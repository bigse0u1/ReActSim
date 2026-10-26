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
