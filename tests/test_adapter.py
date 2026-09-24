"""Phase 3 modules on CPU with tiny random Wan transformers: shapes, zero-init invariants, masking, gradients."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.adapters.motion_adapter import BlockAdapter, MotionAdapter, frame_mask, latent_grid
from models.adapters.stack import ConditioningConfig, MovaConditioning
from models.fusion.condition import GLOBAL, ConditionFusion, ConditionTokens
from models.identity.encoder import IdentityEncoder
from models.motion.encoders import LatentFramePool, MotionEncoder, latent_frames
from models.motion.features import tracks_to_inputs

pytest.importorskip("diffusers")

FRAMES, H, W = 9, 32, 32   # 9 = 4*2+1 video frames -> 3 latent frames; 32x32 -> 4x4 latent -> 2x2 tokens


def tiny_wan():
    from diffusers import WanTransformer3DModel

    torch.manual_seed(0)
    return WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=16,
                                 out_channels=16, text_dim=32, freq_dim=32, ffn_dim=32, num_layers=3,
                                 cross_attn_norm=True, qk_norm="rms_norm_across_heads", rope_max_seq_len=32).eval()


def motion_batch(b=1, t=FRAMES, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"body": torch.randn(b, t, 33, 6, generator=g), "body_mask": torch.ones(b, t, dtype=torch.bool),
            "face": torch.randn(b, t, 58, generator=g), "face_mask": torch.ones(b, t, dtype=torch.bool),
            "hands": torch.randn(b, t, 2, 21, 4, generator=g), "hands_mask": torch.ones(b, t, 2, dtype=torch.bool)}


def dit_inputs(b=1, seed=1):
    g = torch.Generator().manual_seed(seed)
    f = latent_frames(FRAMES)
    return dict(hidden_states=torch.randn(b, 16, f, H // 8, W // 8, generator=g), timestep=torch.tensor([500] * b),
                encoder_hidden_states=torch.randn(b, 8, 32, generator=g), return_dict=False)


# --- shapes -------------------------------------------------------------------------------------------

def test_latent_grid_matches_the_transformer_sequence_length():
    assert latent_grid(17, 256, 256) == (5, 16, 16) and latent_grid(FRAMES, H, W) == (3, 2, 2)
    with pytest.raises(ValueError):
        latent_grid(16, 256, 256)
    seen = []
    tr = tiny_wan()
    tr.blocks[0].register_forward_hook(lambda m, a, o: seen.append(o.shape))
    with torch.no_grad():
        out = tr(**dit_inputs())[0]
    f, hp, wp = latent_grid(FRAMES, H, W)
    assert seen[0][1] == f * hp * wp and out.shape == (1, 16, 3, 4, 4)  # (B, 16, 4k+1 -> k+1, H/8, W/8)


def test_latent_frame_pool_follows_wan_vae_rule():
    pool = LatentFramePool(8)
    x = torch.randn(2, 17, 3, 8)
    m = torch.zeros(2, 17, 3, dtype=torch.bool)
    m[:, 5] = True  # present only in frame 5 -> latent frame (5-1)//4+1 = 2
    y, ym = pool(x, m)
    assert y.shape == (2, 5, 3, 8) and ym[:, 2].all() and ym.sum() == 2 * 3
    with pytest.raises(ValueError):
        pool(torch.randn(1, 16, 3, 8), torch.ones(1, 16, 3, dtype=torch.bool))


def test_motion_encoder_tokens_and_masks():
    enc = MotionEncoder(dim=32, body_tokens=4, face_tokens=2, heads=4)
    batch = motion_batch(b=2)
    batch["face_mask"][1] = False                  # no face at all in sample 1
    batch["hands_mask"][0, :, 1] = False           # right hand never visible in sample 0
    cond = enc(batch)
    f, k = 3, 4 + 2 + 2
    assert cond.tokens.shape == (2, f * k, 32) and torch.isfinite(cond.tokens).all()
    assert cond.frame_index.tolist() == [i for i in range(f) for _ in range(k)]
    per_frame = cond.mask.view(2, f, k)
    assert per_frame[0].all(1).tolist() == [False] * f and per_frame[0, :, 7].eq(False).all()  # right-hand slot
    assert not per_frame[1, :, 4:6].any() and per_frame[1, :, :4].all()


def test_motion_inputs_from_saved_track_format():
    t = 9
    rng = np.random.default_rng(0)
    kp = rng.random((t, 33, 4)).astype(np.float32)
    kp[..., 3] = 1
    tracks = {
        "body": {"kp2d": kp, "kp3d_world": rng.normal(0, .3, (t, 33, 3)), "present": np.ones(t, bool)},
        "face": {"blendshapes": rng.random((t, 52)), "head_transform": np.repeat(np.eye(4)[None], t, 0),
                 "present": np.r_[np.ones(t - 2), np.zeros(2)].astype(bool)},
        "hands": {"kp3d_world": rng.normal(0, .05, (t, 2, 21, 3)), "present": np.ones((t, 2), bool)},
    }
    x = tracks_to_inputs(tracks, start=0, num_frames=t)
    assert x["body"].shape == (t, 33, 6) and x["face"].shape == (t, 58) and x["hands"].shape == (t, 2, 21, 4)
    assert x["face_mask"].tolist() == [True] * 7 + [False] * 2 and (x["face"][-2:] == 0).all()
    assert np.allclose(x["face"][0, 52:], [1, 0, 0, 1, 0, 0])  # identity rotation in 6D
    batch = {k: torch.from_numpy(v)[None] for k, v in x.items()}
    assert MotionEncoder(dim=16, heads=2)(batch).tokens.shape == (1, 3 * 8, 16)
    with pytest.raises(ValueError, match="outside"):
        tracks_to_inputs(tracks, start=4, num_frames=9)


def test_identity_encoder_is_view_order_invariant_and_ignores_masked_views():
    torch.manual_seed(0)
    enc = IdentityEncoder(dim=32, tokens=4, patch=8, heads=4).eval()
    views = torch.rand(1, 3, 3, 32, 32)
    with torch.no_grad():
        a = enc(views).tokens
        b = enc(views[:, [2, 0, 1]]).tokens
        extra = torch.cat([views, torch.rand(1, 1, 3, 32, 32)], 1)
        c = enc(extra, torch.tensor([[True, True, True, False]])).tokens
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)
    out = enc(views)
    assert out.tokens.shape == (1, 4, 32) and (out.frame_index == GLOBAL).all()
    with pytest.raises(ValueError, match="at least one valid"):
        enc(views, torch.zeros(1, 3, dtype=torch.bool))


# --- adapter invariants ------------------------------------------------------------------------------

def test_zero_init_adapter_leaves_the_backbone_bit_identical():
    tr = tiny_wan()
    stack = MovaConditioning.for_transformer(tr, ConditioningConfig(cond_dim=32, adapter_dim=16, heads=4,
                                                                    identity_patch=8, use_dense_pose=True))
    with torch.no_grad():
        ref = tr(**dit_inputs())[0]
        stack.adapter.attach(tr)
        batch = motion_batch() | {"reference": torch.rand(1, 2, 3, 32, 32), "pose_video": torch.rand(1, 3, FRAMES, H, W)}
        with stack.conditioned(batch, FRAMES, H, W):
            out = tr(**dit_inputs())[0]
        assert torch.equal(ref, out)  # step 0: MOVA == backbone, bit for bit
        for m in list(stack.adapter.cross.values()) + list(stack.adapter.dense.values()):
            torch.nn.init.normal_(m.out.weight if hasattr(m, "out") else m.weight, std=0.1)
        with stack.conditioned(batch, FRAMES, H, W):
            changed = tr(**dit_inputs())[0]
        assert not torch.allclose(ref, changed)
        assert torch.equal(tr(**dit_inputs())[0], ref)  # outside the context: pure backbone again
        stack.adapter.detach()
        assert not tr.blocks[0]._forward_hooks


def test_cross_attention_is_frame_local_and_identity_is_global():
    torch.manual_seed(0)
    blk = BlockAdapter(hidden_dim=24, cond_dim=8, dim=8, heads=2)
    torch.nn.init.normal_(blk.out.weight, std=0.5)
    grid = (3, 2, 2)
    hidden = torch.randn(1, 12, 24)
    frames = torch.tensor([0, 0, 1, 1, 2, 2, GLOBAL])
    cond = ConditionTokens(torch.randn(1, 7, 8), frames, torch.ones(1, 7, dtype=torch.bool))
    base = blk(hidden, cond, frame_mask(grid, cond))
    moved = ConditionTokens(cond.tokens.clone(), frames, cond.mask)
    moved.tokens[:, 2:4] += 5.0  # change only frame-1 motion tokens
    out = blk(hidden, moved, frame_mask(grid, moved))
    changed = ~torch.isclose(base, out).all(-1)[0]
    assert changed.tolist() == [False] * 4 + [True] * 4 + [False] * 4
    glob = ConditionTokens(cond.tokens.clone(), frames, cond.mask)
    glob.tokens[:, 6] += 5.0  # the global identity token reaches every frame
    assert (~torch.isclose(base, blk(hidden, glob, frame_mask(grid, glob))).all(-1)).all()


def test_frames_without_any_condition_get_zero_residual_not_nan():
    blk = BlockAdapter(24, 8, 8, 2)
    torch.nn.init.normal_(blk.out.weight, std=0.5)
    frames = torch.tensor([0, 0, 1, 1])
    cond = ConditionTokens(torch.randn(1, 4, 8), frames, torch.tensor([[True, True, False, False]]))
    res = blk(torch.randn(1, 8, 24), cond, frame_mask((2, 2, 2), cond))
    assert torch.isfinite(res).all() and res[0, 4:].abs().max() == 0 and res[0, :4].abs().max() > 0


def test_gradients_reach_only_adapter_and_zero_init_trains_output_first():
    tr = tiny_wan()
    for p in tr.parameters():
        p.requires_grad_(False)
    stack = MovaConditioning.for_transformer(tr, ConditioningConfig(cond_dim=32, adapter_dim=16, heads=4,
                                                                    use_identity=False))
    stack.adapter.attach(tr)
    with stack.conditioned(motion_batch(), FRAMES, H, W):
        out = tr(**dit_inputs())[0]
    out.pow(2).mean().backward()
    assert all(p.grad is None for p in tr.parameters())
    outs = [b.out.weight.grad for b in stack.adapter.cross.values()]
    assert all(g is not None and g.abs().sum() > 0 for g in outs)
    enc_grads = [p.grad for p in stack.motion.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in enc_grads)  # zero-init: encoders start learning after out moves


def test_condition_batch_broadcasts_for_cfg_and_rejects_mismatch():
    tr = tiny_wan()
    stack = MovaConditioning.for_transformer(tr, ConditioningConfig(cond_dim=32, adapter_dim=16, heads=4,
                                                                    use_identity=False))
    stack.adapter.attach(tr)
    for m in stack.adapter.cross.values():
        torch.nn.init.normal_(m.out.weight, std=0.1)
    with torch.no_grad(), stack.conditioned(motion_batch(b=1), FRAMES, H, W):
        out = tr(**dit_inputs(b=2))[0]  # cond + uncond batched
    assert out.shape[0] == 2
    with pytest.raises(ValueError, match="grid"), torch.no_grad(), stack.conditioned(motion_batch(), 13, H, W):
        tr(**dit_inputs())


def test_fusion_adds_type_embedding_and_keeps_masks():
    fuse = ConditionFusion(4)
    torch.nn.init.normal_(fuse.type_embed)
    m = ConditionTokens(torch.zeros(1, 3, 4), torch.tensor([0, 0, 1]), torch.tensor([[True, False, True]]))
    i = ConditionTokens(torch.zeros(1, 2, 4), torch.tensor([GLOBAL, GLOBAL]), torch.ones(1, 2, dtype=torch.bool))
    out = fuse(motion=m, identity=i)
    assert out.tokens.shape == (1, 5, 4) and out.mask.tolist() == [[True, False, True, True, True]]
    assert torch.allclose(out.tokens[0, 0], fuse.type_embed[0]) and torch.allclose(out.tokens[0, 4], fuse.type_embed[1])
    with pytest.raises(ValueError):
        ConditionTokens(torch.zeros(1, 3, 4), torch.tensor([0]), torch.ones(1, 3, dtype=torch.bool))


# --- end to end through the real VACE pipeline --------------------------------------------------------

def test_zero_init_adapter_in_the_vace_pipeline_reproduces_the_phase0_baseline():
    """Attach MOVA conditioning to the tiny VACE transformer: generated frames keep the baseline SHA-256."""
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "benchmark/baseline"))
    from capture_tiny_vace import FRAMES as F0, H as H0, W as W0, frames_digest, synthetic_inputs, tiny_vace_pipe

    from inference.baseline_vace import BaselineSettings, generate

    expected = json.loads((root / "benchmark/baseline/tiny_vace_cpu.json").read_text())["runs"][0]["output"]["sha256"]
    pipe = tiny_vace_pipe()
    stack = MovaConditioning.for_transformer(pipe.transformer, ConditioningConfig(cond_dim=32, adapter_dim=16, heads=4,
                                                                                  identity_patch=8))
    stack.adapter.attach(pipe.transformer)
    ref, control, pe, ne = synthetic_inputs()
    s = BaselineSettings(height=H0, width=W0, num_frames=F0, num_inference_steps=2, dtype="float32", seed=42)
    batch = motion_batch(t=F0) | {"reference": torch.rand(1, 1, 3, 32, 32)}
    with torch.no_grad(), stack.conditioned(batch, F0, H0, W0, reference_frames=1):  # VACE prepends the reference
        frames, _ = generate(pipe, s, ref, control, pe, ne)
    assert frames_digest(frames)["sha256"] == expected
    for m in stack.adapter.cross.values():  # the same wiring really changes the video once trained
        torch.nn.init.normal_(m.out.weight, std=0.2)
    with torch.no_grad(), stack.conditioned(batch, F0, H0, W0, reference_frames=1):
        changed, _ = generate(pipe, s, ref, control, pe, ne)
    assert frames_digest(changed)["sha256"] != expected


def test_reference_frames_only_see_identity_tokens():
    frames = torch.tensor([0, 1, GLOBAL])
    cond = ConditionTokens(torch.zeros(1, 3, 4), frames, torch.ones(1, 3, dtype=torch.bool))
    allowed = frame_mask((3, 1, 1), cond, reference_frames=1)[0]   # latent frames: [ref, video0, video1]
    assert allowed.tolist() == [[False, False, True], [True, False, True], [False, True, True]]
