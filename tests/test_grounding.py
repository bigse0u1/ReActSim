import numpy as np

from reactsim.perception.grounding import backproject_mask


def test_backproject_recovers_world_point():
    K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])
    # camera 1 m above origin looking straight down (OpenCV: z forward, y down)
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    C = np.array([0.0, 0.0, 1.0])
    ext = np.hstack([R, (-R @ C)[:, None]])
    target = np.array([0.1, -0.05, 0.0])
    pc = R @ target + ext[:, 3]
    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    depth = np.zeros((64, 64))
    mask = np.zeros((64, 64), bool)
    ui, vi = int(round(u)), int(round(v))
    mask[vi - 3:vi + 3, ui - 3:ui + 3] = True
    depth[mask] = pc[2]
    p = backproject_mask(mask, depth, K, ext)
    assert np.allclose(p, target, atol=0.01)


def test_estimate_box_from_visible_half():
    """The camera sees only the near half of a can plus its top: the box centre must still
    land near the true axis (the plain centroid would be pulled ~1.5 cm toward the camera)."""
    from reactsim.perception.grounding import estimate_box

    r, h, c = 0.033, 0.10, np.array([0.1, -0.2])
    side = np.array([[c[0] + r * np.cos(t), c[1] + r * np.sin(t), z]
                     for t in np.linspace(-np.pi / 2, np.pi / 2, 60) for z in np.linspace(0.005, h, 30)])
    top = np.array([[c[0] + q * np.cos(t), c[1] + q * np.sin(t), h]
                    for t in np.linspace(0, 2 * np.pi, 40) for q in np.linspace(0, r, 10)])
    box = estimate_box(np.vstack([side, top]))
    centre = np.array(box.primitive.transform)[:3, 3]
    assert np.linalg.norm(centre[:2] - c) < 0.012
    assert abs(box.primitive.extents[2] - h) < 0.005


def test_estimate_box_long_axis():
    from reactsim.perception.grounding import estimate_box

    yaw = np.deg2rad(30)
    s = np.linspace(-0.09, 0.09, 80)
    pts = np.array([[t * np.cos(yaw) + d * -np.sin(yaw), t * np.sin(yaw) + d * np.cos(yaw), z]
                    for t in s for d in (-0.015, 0, 0.015) for z in (0.01, 0.03)])
    axis = np.array(estimate_box(pts).primitive.transform)[:3, 0]
    assert abs(abs(axis[:2] @ [np.cos(yaw), np.sin(yaw)]) - 1) < 1e-3
