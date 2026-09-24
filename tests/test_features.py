import numpy as np

from preprocessing import features as F
from preprocessing.render import blank, draw_openpose_body, mp_body_to_openpose18
from preprocessing.topology import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER


def _body(T=4, scale=1.0, offset=(0.0, 0.0)):
    kp = np.random.default_rng(0).uniform(0.3, 0.7, size=(T, 33, 2)).astype(np.float32)
    kp[:, LEFT_SHOULDER] = [0.45, 0.3]
    kp[:, RIGHT_SHOULDER] = [0.55, 0.3]
    kp[:, LEFT_HIP] = [0.45, 0.6]
    kp[:, RIGHT_HIP] = [0.55, 0.6]
    return kp * scale + np.array(offset, np.float32)


def test_interpolate_missing_linear():
    x = np.array([[0.0], [0.0], [0.0], [3.0]], np.float32)[:, None, :]
    present = np.array([True, False, False, True])
    out = F.interpolate_missing(x, present)
    np.testing.assert_allclose(out[:, 0, 0], [0, 1, 2, 3])


def test_velocity_and_acceleration():
    x = (np.arange(5, dtype=np.float32) ** 2)[:, None, None]
    v = F.velocity(x, fps=10)
    a = F.acceleration(x, fps=10)
    np.testing.assert_allclose(v[1:, 0, 0], np.diff(x[:, 0, 0]) * 10)
    np.testing.assert_allclose(a[2:, 0, 0], 200.0)


def test_normalize_body_invariant_to_scale_and_translation():
    a, _, sa = F.normalize_body(_body())
    b, _, sb = F.normalize_body(_body(scale=2.0, offset=(0.1, -0.2)))
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(sb, 2 * sa, rtol=1e-5)


def test_normalize_hand_unit_scale():
    kp = np.zeros((2, 21, 3), np.float32)
    kp[:, 9] = [0, 2, 0]
    out = F.normalize_hand(kp)
    np.testing.assert_allclose(np.linalg.norm(out[:, 9], axis=-1), 1.0)


def test_rotation_conversions():
    R = np.eye(3, dtype=np.float32)[None]
    np.testing.assert_allclose(F.rotation_matrix_to_euler(R), 0, atol=1e-5)
    th = np.radians(30)
    Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], np.float32)[None]
    np.testing.assert_allclose(F.rotation_matrix_to_euler(Ry)[0, 1], 30, atol=1e-3)
    assert F.rotation_matrix_to_6d(R).shape == (1, 6)


def test_ema_and_jitter():
    rng = np.random.default_rng(1)
    noisy = rng.normal(size=(50, 5, 2)).astype(np.float32)
    assert F.temporal_jitter(F.ema_smooth(noisy, 0.3)) < F.temporal_jitter(noisy)


def test_body_preview_uses_visibility_not_depth():
    from preprocessing.render import body_xyv, draw_mp_body

    kp = np.zeros((33, 4), np.float32)
    kp[:, :2] = _body(1)[0]
    kp[:, 2] = -0.3  # negative relative depth (normal for MediaPipe) must not hide points
    kp[:, 3] = 0.99
    assert draw_mp_body(blank(128, 128), body_xyv(kp)).sum() > 0


def test_openpose_mapping_and_render():
    kp = np.zeros((33, 3), np.float32)
    kp[:, :2] = _body(1)[0]
    kp[:, 2] = 1.0
    op = mp_body_to_openpose18(kp)
    np.testing.assert_allclose(op[1, :2], (kp[11, :2] + kp[12, :2]) / 2)
    np.testing.assert_allclose(op[2, :2], kp[12, :2])  # OpenPose RShoulder = MediaPipe right_shoulder
    img = draw_openpose_body(blank(128, 128), op)
    assert img.sum() > 0
