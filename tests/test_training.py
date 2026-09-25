"""Training scaffolding on CPU: flow-matching loss, region weights, latents precompute, dataset validation,
trainer (overfit, frozen backbone, exact resume) and the `mova train` service."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from models.adapters.stack import ConditioningConfig, MovaConditioning  # noqa: E402
from training.dataset import MotionClipDataset, collate  # noqa: E402
from training.losses import (flow_matching_loss, flow_matching_pair, keypoint_region_mask, region_weights,  # noqa: E402
                             sample_sigmas, shift_sigmas)
from training.smoke import make_synthetic_dataset, tiny_transformer  # noqa: E402
from training.trainer import AdapterTrainer, TrainConfig  # noqa: E402

SMALL = ConditioningConfig(cond_dim=32, adapter_dim=16, heads=4, identity_patch=8)


# --- losses ---------------------------------------------------------------------------------------

def test_flow_matching_pair_endpoints_and_target():
    x0, noise = torch.randn(2, 16, 3, 4, 4), torch.randn(2, 16, 3, 4, 4)
    xt, v = flow_matching_pair(x0, noise, torch.tensor([0.0, 1.0]))
    assert torch.allclose(xt[0], x0[0]) and torch.allclose(xt[1], noise[1]) and torch.allclose(v, noise - x0)
    assert flow_matching_loss(v, v) == 0


def test_sigma_shift_matches_wan_and_stays_in_range():
    u = torch.linspace(0.01, 0.99, 50)
    s = shift_sigmas(u, 3.0)
    assert torch.all(s > u) and torch.all(torch.diff(s) > 0) and s.max() < 1  # shift 3 favours high noise
    g = torch.Generator().manual_seed(0)
    assert torch.all((sample_sigmas(1000, 3.0, g) > 0) & (sample_sigmas(1000, 3.0, g) < 1))


def test_region_mask_pools_video_frames_like_the_vae():
    pts = torch.full((1, 9, 2, 2), 0.5)
    valid = torch.zeros(1, 9, 2, dtype=torch.bool)
    valid[0, 6] = True                                   # visible only in video frame 6 -> latent frame 2
    pts[0, 6] = torch.tensor([[0.05, 0.05], [0.35, 0.35]])  # box over the 3x3 top-left cell centres
    m = keypoint_region_mask(pts, valid, (8, 8), 3, pad=0.0)
    assert m.shape == (1, 3, 8, 8) and m[0, :2].sum() == 0 and m[0, 2, :3, :3].all() and m[0, 2].sum() == 9


def test_region_weights_upweight_regions_and_keep_mean_one():
    mask = torch.zeros(2, 3, 4, 4)
    mask[0, :, :2, :2] = 1
    w = region_weights((2, 16, 3, 4, 4), [(mask, 3.0)])
    assert w.shape == (2, 1, 3, 4, 4)
    assert torch.allclose(w.mean(dim=(1, 2, 3, 4)), torch.ones(2))
    assert w[0, 0, 0, 0, 0] == pytest.approx(3 * w[0, 0, 0, 3, 3]) and torch.allclose(w[1], torch.ones_like(w[1]))


# --- precompute --------------------------------------------------------------------------------------

def test_latent_precompute_matches_pipeline_normalisation():
    from diffusers import AutoencoderKLWan

    from training.precompute import encode_latents

    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           temperal_downsample=[False, True, True]).eval()
    frames = [np.random.default_rng(i).integers(0, 255, (32, 32, 3), dtype=np.uint8) for i in range(9)]
    z = encode_latents(vae, frames)
    assert z.shape == (16, 3, 4, 4) and torch.isfinite(z).all()
    raw = vae.encode(torch.from_numpy(np.stack(frames)).float().div(127.5).sub(1).permute(3, 0, 1, 2)[None]).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean).view(1, 16, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 16, 1, 1, 1)
    assert torch.allclose(z, ((raw - mean) * inv_std)[0], atol=1e-5)   # same formula as WanVACEPipeline
    with pytest.raises(ValueError, match="4k\\+1"):
        encode_latents(vae, frames[:8])


# --- dataset -------------------------------------------------------------------------------------

def test_dataset_loads_synthetic_manifest(tmp_path):
    ds = MotionClipDataset(make_synthetic_dataset(tmp_path, 2), reference_size=32, max_references=2)
    b = collate([ds[0], ds[1]])
    assert b["latents"].shape == (2, 16, 3, 4, 4) and b["body"].shape == (2, 9, 33, 6)
    assert b["reference"].shape == (2, 2, 3, 32, 32) and b["reference_mask"].tolist() == [[True, False]] * 2
    assert b["face_xy"].shape == (2, 9, 60, 2) and b["hands_xy_mask"].shape == (2, 9, 42)


def test_dataset_rejects_broken_samples_and_benchmark_identities(tmp_path):
    manifest = make_synthetic_dataset(tmp_path, 2)
    data = json.loads(manifest.read_text())
    with pytest.raises(ValueError, match="reserved for evaluation"):
        MotionClipDataset(manifest, forbid_identities={"synthetic-1"})
    data["samples"][0]["latents"] = "clip0/missing.pt"
    data["samples"][1]["num"] = 1
    del data["samples"][1]["identity_id"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError) as e:
        MotionClipDataset(manifest)
    assert "files not found" in str(e.value) and "missing ['identity_id']" in str(e.value)
    with pytest.raises(ValueError, match="No 'val' samples"):
        MotionClipDataset(make_synthetic_dataset(tmp_path / "b", 1), split="val")


def test_dataset_rejects_wrong_latent_shape(tmp_path):
    manifest = make_synthetic_dataset(tmp_path, 1)
    torch.save(torch.zeros(16, 2, 4, 4), tmp_path / "clip0" / "latents.pt")
    with pytest.raises(ValueError, match="expected \\(16, 3, 4, 4\\)"):
        MotionClipDataset(manifest)


# --- trainer ------------------------------------------------------------------------------------

def _trainer(tmp_path, backbone="tiny", lr=3e-3, **train):
    from training.backbone import load_backbone

    tr, extra = load_backbone(backbone)
    cond = MovaConditioning.for_transformer(tr, SMALL)
    ds = MotionClipDataset(make_synthetic_dataset(tmp_path, 2), reference_size=32, max_references=2)
    return AdapterTrainer(tr, cond, TrainConfig(lr=lr, seed=0, **train), forward_extra=extra), collate([ds[0], ds[1]])


def test_gradient_checkpointing_matches_plain_backward(tmp_path):
    """Checkpointing recomputes blocks during backward: the adapter hooks must still be live then (else torch
    raises 'different number of tensors saved'), and the gradients must equal the plain ones."""
    sigma = torch.full((2,), 0.5)
    grads = []
    for ckpt in (False, True):
        t, batch = _trainer(tmp_path / str(ckpt), backbone="tiny-vace", gradient_checkpointing=ckpt, grad_accum=2)
        noise = torch.randn(batch["latents"].shape, generator=torch.Generator().manual_seed(1))
        t.train_step(batch, sigma=sigma, noise=noise)  # accumulates, no optimizer step yet
        grads.append(torch.cat([p.grad.flatten() for p in t.params if p.grad is not None]))
    assert grads[0].abs().sum() > 0
    assert torch.allclose(grads[0], grads[1], atol=1e-6, rtol=1e-4)


def test_overfits_a_fixed_batch_and_never_touches_the_backbone(tmp_path):
    t, batch = _trainer(tmp_path)
    before = {k: v.clone() for k, v in t.transformer.state_dict().items()}
    sigma, noise = torch.full((2,), 0.5), torch.randn(batch["latents"].shape, generator=torch.Generator().manual_seed(1))
    losses = [t.train_step(batch, sigma=sigma, noise=noise)["loss"] for _ in range(30)]
    assert losses[-1] < 0.85 * losses[0]
    assert all(torch.equal(before[k], v) for k, v in t.transformer.state_dict().items())


def test_resume_continues_exactly(tmp_path):
    a, batch = _trainer(tmp_path / "a")
    for _ in range(3):
        a.train_step(batch)
    ck = a.save(tmp_path / "ck.pt")
    cont = [a.train_step(batch)["loss"] for _ in range(2)]
    b, _ = _trainer(tmp_path / "b")
    b.load(ck)
    assert b.step_count == 3
    resumed = [b.train_step(batch)["loss"] for _ in range(2)]
    assert resumed == pytest.approx(cont, rel=1e-6, abs=1e-7)
    other = MovaConditioning.for_transformer(tiny_transformer(), ConditioningConfig(cond_dim=16, adapter_dim=16, heads=4))
    with pytest.raises(ValueError, match="different conditioning config"):
        AdapterTrainer(tiny_transformer(), other).load(ck)


def test_grad_accumulation_updates_every_n_micro_batches(tmp_path):
    t, batch = _trainer(tmp_path)
    t.cfg.grad_accum = 3
    flags = [t.train_step(batch)["updated"] for _ in range(6)]
    assert flags == [False, False, True, False, False, True] and t.step_count == 2


def test_vace_backbone_trains_with_null_control(tmp_path):
    t, batch = _trainer(tmp_path, backbone="tiny-vace")
    rec = t.train_step(batch)
    assert np.isfinite(rec["loss"]) and rec["updated"]


def test_real_backbone_is_never_downloaded(monkeypatch):
    """The trainer only reads the local cache; a cache miss is a clear error, never a download.
    from_pretrained is stubbed so the test holds (and stays light) on machines that do have the weights."""
    from diffusers import WanVACETransformer3DModel

    from common.errors import ModelWeightsMissingError
    from training.backbone import load_backbone

    calls = []

    def cache_miss(*args, **kwargs):
        calls.append(kwargs)
        raise OSError("not cached")

    monkeypatch.setattr(WanVACETransformer3DModel, "from_pretrained", cache_miss)
    with pytest.raises(ModelWeightsMissingError, match="not in the local cache"):
        load_backbone("wan2.1-vace-1.3b")
    assert calls and calls[0]["local_files_only"] is True


# --- service -----------------------------------------------------------------------------------

def test_train_service_smoke_records_the_run(tmp_path):
    from core.train import TrainRequest, run_training

    res = run_training(TrainRequest(smoke=True, steps=4, device="cpu", runs_dir=tmp_path / "runs"), echo=lambda _: None)
    rec = json.loads(open(res["record"]).read())
    assert res["steps"] == 4 and rec["status"] == "success" and rec["optimizer"] == "AdamW"
    assert rec["learning_rate"] == 3e-3 and rec["loss"] == res["last_loss"] and rec["trainable_parameters"] > 0
    assert rec["runtime"]["device"] == "cpu"


def test_benchmark_identities_are_collected(tmp_path):
    from core.train import benchmark_identities

    y = tmp_path / "b.yaml"
    y.write_text("cases:\n- {id: a, identity_id: person-1}\n- {id: b, identity_id: null}\n")
    assert benchmark_identities(y) == {"person-1"}
