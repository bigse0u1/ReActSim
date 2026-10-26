"""Phase 0 smoke test: ManiSkill + custom env load, step, render, and camera params exist."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    import gymnasium as gym
    import numpy as np

    import reactsim.envs.tabletop  # noqa: F401

    env = gym.make("ReActTabletop-v1", obs_mode="rgb+depth+segmentation",
                   control_mode="pd_joint_pos", render_mode="rgb_array", num_envs=1)
    obs, _ = env.reset(seed=0)
    cam = obs["sensor_data"]["base_camera"]
    print("rgb", tuple(cam["rgb"].shape), "depth", tuple(cam["depth"].shape))
    print("intrinsic_cv\n", np.round(obs["sensor_param"]["base_camera"]["intrinsic_cv"][0].cpu().numpy(), 2))
    print("seg ids", {k: int(a.per_scene_id[0]) for k, a in env.unwrapped.objects.items()})
    for _ in range(50):
        obs, *_ = env.step(env.action_space.sample())
    print("50 random steps OK")
    env.close()


if __name__ == "__main__":
    main()
