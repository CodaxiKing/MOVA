import copy
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import yaml

from common.experiment import ExperimentRun
from common.video_io import write_video
from evaluation.benchmark import compare_reports, evaluate_benchmark
from evaluation.motion import compare_tracks
from evaluation.protocol import freeze_manifest, inspect_manifest, read_manifest, verify_lock
from preprocessing.mp_models import MODEL_DIR
from scripts.benchmark import generation_config, prepare_video


def tracks(n=5):
    meta = dict(format_version=1, num_frames=n, fps=16, width=64, height=32)
    body = np.ones((n, 33, 4))
    body[..., :3] = 0.5
    body[:, [11, 12], 1] = 0.2
    hands = np.ones((n, 2, 21, 3)) * 0.5
    hands[:, :, 9, 1] = 0.6
    return {
        'body': dict(meta=meta.copy(), kp2d=body, present=np.ones(n, bool)),
        'hands': dict(meta=meta.copy(), kp2d=hands, present=np.ones((n, 2), bool)),
        'face': dict(meta=meta.copy(), blendshapes=np.zeros((n, 52)),
                     present=np.ones(n, bool), blendshape_names=[str(i) for i in range(52)])}


def test_metrics_identical_and_missing():
    reference = tracks()
    metrics = compare_tracks(reference, copy.deepcopy(reference))
    assert metrics['body']['pck'] == 1
    assert metrics['hands']['pck'] == 1
    assert metrics['face']['blendshape_mae_paired'] == 0
    assert metrics['body']['acceleration_error_paired'] == 0
    generated = copy.deepcopy(reference)
    for value in generated.values():
        value['present'][:] = False
    missing = compare_tracks(reference, generated)
    assert missing['body']['pck'] == 0
    assert missing['hands']['mean_error_paired'] is None
    assert missing['face']['blendshape_mae_paired'] is None
    assert missing['body']['acceleration_error_paired'] is None
    absent = compare_tracks(generated, generated)
    assert absent['body']['pck'] is None
    json.dumps(absent, allow_nan=False)


def test_metrics_known_error_and_aspect_ratio():
    reference = tracks()
    generated = copy.deepcopy(reference)
    # Torso = 9.6 pixels. Offset one point by 0.05*64 = 3.2 pixels.
    generated['body']['kp2d'][:, 0, 0] += 0.05
    result = compare_tracks(reference, generated)['body']
    assert result['pck'] == pytest.approx(32/33)
    assert result['mean_error_paired'] == pytest.approx((3.2/9.6)/33)
    # Same pixel geometry in an image with a different aspect ratio.
    generated = copy.deepcopy(reference)
    generated['body']['meta']['width'] = 128
    generated['body']['kp2d'][..., 0] /= 2
    assert compare_tracks(reference, generated)['body']['mean_error_paired'] == pytest.approx(0)


def test_missing_anchors_and_nonfinite_are_not_perfect():
    reference = tracks()
    generated = copy.deepcopy(reference)
    generated['body']['kp2d'][:, 11, 3] = 0
    assert compare_tracks(reference, generated)['body']['pck'] == 0
    generated = copy.deepcopy(reference)
    generated['hands']['kp2d'][:, :, 0] = np.nan
    assert compare_tracks(reference, generated)['hands']['pck'] == 0


def test_temporal_error_and_missing_gaps():
    reference = tracks()
    generated = copy.deepcopy(reference)
    generated['body']['kp2d'][2, 0, 0] += 0.1
    assert compare_tracks(reference, generated)['body']['acceleration_error_paired'] > 0
    generated['body']['present'][2] = False
    assert compare_tracks(reference, generated)['body']['acceleration_error_paired'] is None


@pytest.mark.parametrize('key,value', [('fps', 8), ('num_frames', 4), ('format_version', 2), ('format_version', 3)])
def test_alignment_is_strict(key, value):
    reference, generated = tracks(), tracks()
    generated['face']['meta'][key] = value
    with pytest.raises(ValueError):
        compare_tracks(reference, generated)


def test_blendshape_order_rejected():
    reference, generated = tracks(), tracks()
    generated['face']['blendshape_names'].reverse()
    with pytest.raises(ValueError, match='ordering'):
        compare_tracks(reference, generated)


def manifest_with_media(tmp_path):
    from PIL import Image

    image = tmp_path / 'ref.png'
    Image.new('RGB', (32, 32)).save(image)
    video = write_video(tmp_path / 'motion.mp4', np.zeros((5, 32, 32, 3), np.uint8), 16)
    return dict(schema_version=1, version='synthetic-test-only',
                protocol=dict(width=32, height=32, frames=5, fps=16, seed=42, pck_threshold=0.1, visibility=0.5),
                cases=[dict(id='test-01', category='synthetic', split='test', identity_id='none',
                            prompt='Synthetic test', reference=str(image), motion=str(video),
                            reference_source='test fixture', motion_source='test fixture',
                            reference_license='generated fixture', motion_license='generated fixture')])


def test_freeze_detects_file_and_lock_tampering(tmp_path):
    manifest = manifest_with_media(tmp_path)
    lock = tmp_path / 'lock.json'
    frozen = freeze_manifest(manifest, lock)
    assert verify_lock(lock) == frozen
    with pytest.raises(FileExistsError):
        freeze_manifest(manifest, lock)
    changed = copy.deepcopy(frozen)
    changed['protocol']['seed'] = 43
    lock.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='modified'):
        verify_lock(lock)
    lock.write_text(json.dumps(frozen))
    with open(manifest['cases'][0]['motion'], 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        verify_lock(lock)


def test_incomplete_manifest_cannot_freeze(tmp_path):
    manifest = manifest_with_media(tmp_path)
    manifest['cases'][0]['motion_license'] = None
    assert inspect_manifest(manifest)
    with pytest.raises(ValueError, match='incomplete'):
        freeze_manifest(manifest, tmp_path / 'bad.json')


def test_manifest_duplicate_and_generation_contract(tmp_path):
    manifest = manifest_with_media(tmp_path)
    cfg = generation_config(manifest, manifest['cases'][0])
    assert cfg['generation']['keep_reference_aspect'] is False
    assert cfg['generation']['num_frames'] == 5
    assert cfg['inputs']['frame_stride'] == 1
    path = tmp_path / 'manifest.yaml'
    manifest['cases'] *= 2
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match='duplicate'):
        read_manifest(path)


def test_prepare_rejects_short_video_and_roundtrips(tmp_path):
    manifest = manifest_with_media(tmp_path)
    source = manifest['cases'][0]['motion']
    p = manifest['protocol']
    assert prepare_video(source, tmp_path / 'prepared.mp4', p)['num_frames'] == 5
    with pytest.raises(ValueError, match='too short'):
        prepare_video(source, tmp_path / 'short.mp4', {**p, 'frames': 9})
    assert not (tmp_path / 'short.mp4').exists()
    with pytest.raises(ValueError, match='FPS'):
        prepare_video(source, tmp_path / 'fps.mp4', {**p, 'fps': 30})


def test_comparison_requires_matching_conditions():
    report = dict(schema_version=1, benchmark_sha256='a', evaluator_sha256='b', protocol={},
                  generation_contract={'seed': 42}, status='complete',
                  cases=[dict(id='one', status='evaluated', metrics=compare_tracks(tracks(), tracks()))])
    compared = compare_reports(report, copy.deepcopy(report))
    assert compared['status'] == 'REQUIRES_REVIEW'
    assert compared['cases'][0]['candidate_minus_baseline']['body.pck'] == 0
    for key in ('benchmark_sha256', 'evaluator_sha256', 'generation_contract', 'status'):
        changed = copy.deepcopy(report)
        changed[key] = 'different'
        with pytest.raises(ValueError):
            compare_reports(report, changed)


@pytest.mark.skipif(not all((MODEL_DIR / f'{n}.task').exists() for n in
                           ('pose_landmarker_full', 'face_landmarker', 'hand_landmarker')),
                    reason='Requires cached MediaPipe weights')
@pytest.mark.parametrize('person', [False, True])
def test_real_benchmark_evaluation(tmp_path, monkeypatch, person):
    import evaluation.benchmark as benchmark

    monkeypatch.setattr(benchmark, 'ExperimentRun', partial(ExperimentRun, runs_dir=tmp_path / 'runs'))
    manifest = manifest_with_media(tmp_path)
    if person:
        from skimage.data import astronaut
        from PIL import Image

        image = np.asarray(Image.fromarray(astronaut()).resize((256, 256)))
        manifest['protocol'].update(width=256, height=256)
        write_video(manifest['cases'][0]['motion'], [image] * 5, 16)
    lock = tmp_path / 'lock.json'
    freeze_manifest(manifest, lock)
    outputs = tmp_path / 'generated'
    outputs.mkdir()
    (outputs / 'test-01.mp4').write_bytes(Path(manifest['cases'][0]['motion']).read_bytes())
    path, report = evaluate_benchmark(lock, outputs, label='synthetic-only', generation_contract={'seed': 42},
                                      artifacts_root=tmp_path / 'artifacts')
    assert report['status'] == 'complete'
    metrics = report['cases'][0]['metrics']
    assert metrics['identity']['metric_version'] == 'identity-v1'
    assert metrics['temporal']['metric_version'] == 'temporal-v1' and metrics['temporal']['generated']['pairs'] == 4
    review = (path.parent / 'review' / 'index.html').read_text(encoding='utf-8')
    assert 'Identity' in review and 'Temporal (dynamic quality)' in review
    if person:
        # The portrait lacks visible hip anchors: do not weaken the metric to score it.
        assert metrics['body']['pck'] is None
        assert metrics['face']['paired_coverage'] > 0.75
        assert metrics['face']['blendshape_mae_paired'] == pytest.approx(0)
        # Repeat the same evaluation: identical inputs must remain comparable.
        _, repeated = evaluate_benchmark(lock, outputs, label='repeat', generation_contract={'seed': 42},
                                          artifacts_root=tmp_path / 'artifacts')
        compared = compare_reports(report, repeated)
        assert compared['cases'][0]['candidate_minus_baseline']['face.blendshape_mae_paired'] == pytest.approx(0)
        assert compared['cases'][0]['candidate_minus_baseline']['temporal.generated.warp_error'] == pytest.approx(0)
    else:
        assert metrics['body']['pck'] is None
    assert json.loads(path.read_text())['quality_status'] == 'NOT_ESTABLISHED'
    _, missing = evaluate_benchmark(lock, tmp_path / 'missing', label='missing', generation_contract={'seed': 42},
                                    artifacts_root=tmp_path / 'artifacts')
    assert missing['status'] == 'partial'
    assert len(missing['cases']) == 1


def test_generate_cli_delegates_and_records_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.benchmark as cli

    manifest = manifest_with_media(tmp_path)
    lock = tmp_path / 'lock.json'
    freeze_manifest(manifest, lock)
    monkeypatch.setattr(cli, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(cli, 'detect_hardware', lambda: SimpleNamespace(cuda_available=True))
    monkeypatch.setattr(cli.sys, 'argv', ['benchmark.py', 'generate', '--lock', str(lock)])

    def fake_baseline(command, **kwargs):
        assert '--allow-download' not in command
        cfg = yaml.safe_load(Path(command[-1]).read_text())
        output = Path(cfg['output']['dir']) / 'test-run'
        output.mkdir(parents=True)
        (output / 'output.mp4').write_bytes(Path(manifest['cases'][0]['motion']).read_bytes())
        record = tmp_path / 'experiments/runs/test-run'
        record.mkdir(parents=True)
        (record / 'run.json').write_text(json.dumps(dict(stats=dict(generation_time_s=2, vram_peak_gb=None),
                                                       hardware={}, settings=cfg['generation'], git_commit='test')))

    monkeypatch.setattr(cli.subprocess, 'run', fake_baseline)
    assert cli.main() == 0
    root = next((tmp_path / 'outputs/benchmark').iterdir())
    index = json.loads((root / 'generation.json').read_text())
    assert index['cases']['test-01']['seconds_per_frame'] == 0.4
    assert (root / 'videos/test-01.mp4').exists()
    monkeypatch.setattr(cli, 'detect_hardware', lambda: SimpleNamespace(cuda_available=False))
    with pytest.raises(ValueError, match='requires CUDA'):
        cli.main()


def test_format_v2_tracks_compare_with_each_other():
    reference, generated = tracks(), tracks()
    for t in (reference, generated):
        for kind in ('body', 'hands', 'face'):
            t[kind]['meta']['format_version'] = 2
    assert compare_tracks(reference, generated)['body']['paired_coverage'] is not None
