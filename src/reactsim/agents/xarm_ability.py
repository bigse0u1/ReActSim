"""xArm7 + PSYONIC Ability hand: a five-finger body for the same agent.

ManiSkill ships this robot (`xarm7_ability`), but as shipped it does not simulate or
plan reliably here. Each fix below was found by measuring:

- Finger q2 joints mimic q1 through PhysX gear constraints, which blow up (NaN) as soon
  as the arm accelerates. The mimic tags are removed and q2 is driven directly
  (q2 = MIMIC_K * q1 + MIMIC_B), and the scene runs at 500 Hz (HAND_SIM_FREQ).
- The upstream config misspells "thumnb_L2", so the thumb pad keeps friction 0.3
  while the fingers get 2.0. All five pads get the high-friction material here.
- The arm PD controller sags centimetres under gravity; real arms compensate it, so
  gravity is off for the robot links and the arm is stiffer.
- mplib needs convex collision meshes (absent), treats overlapping finger hulls as
  self-collision (so every IK solution is rejected), and with +-2pi joint limits picks
  paths that spin joint 1 a full turn into the table. The planning model gets convex
  hulls, finger-finger collisions disabled and joints 1/3/5 limited to +-pi.

The patched files are generated once into ~/.cache/reactsim/robots.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
from pathlib import Path

import numpy as np
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.xarm.xarm7_ability import XArm7Ability

UID = "xarm7_ability_hand"
BASE_X = -0.40            # closer than the Panda (-0.615): the xArm7 reaches less
HAND_SIM_FREQ = 500       # finger links are tiny; at 100 Hz they explode
MIMIC_K, MIMIC_B = 1.05851325, 0.72349796
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
CACHE = Path(os.environ.get("REACTSIM_CACHE", Path.home() / ".cache" / "reactsim")) / "robots"
URDF = "xarm7/xarm7_ability_right_hand.urdf"


def build_assets() -> dict[str, Path]:
    """Write the simulation and planning models (idempotent). Returns their paths."""
    sim = CACHE / "xarm7/xarm7_ability_hand.urdf"
    plan = CACHE / "xarm7/xarm7_ability_hand_plan.urdf"
    srdf = CACHE / "xarm7/xarm7_ability_hand_plan.srdf"
    if not (sim.exists() and plan.exists() and srdf.exists()):
        import trimesh

        CACHE.mkdir(parents=True, exist_ok=True)
        for d in ("xarm7", "ability_hand"):
            shutil.copytree(Path(PACKAGE_ASSET_DIR) / "robots" / d, CACHE / d, dirs_exist_ok=True)
        text = (CACHE / URDF).read_text()
        for f in sorted(set(re.findall(r'<collision>.*?filename="([^"]+)"', text, re.S))):
            mesh = (CACHE / "xarm7" / f).resolve()
            hull = Path(str(mesh) + ".convex.stl")
            if not hull.exists():
                trimesh.load(mesh, force="mesh").convex_hull.export(hull)
        no_mimic = re.sub(r"<mimic [^>]*/>", "", text)
        sim.write_text(no_mimic)

        def narrow(m):
            return re.sub(r'lower="[^"]+" upper="[^"]+"', 'lower="-3.1" upper="3.1"', m.group(0))
        plan.write_text(re.sub(r'<joint name="joint[135]" type="revolute">.*?</joint>', narrow, no_mimic, flags=re.S))
        links = [n for n in re.findall(r'<link name="([^"]+)"', text) if n.split("_")[0] in FINGERS]
        extra = "".join(f'  <disable_collisions link1="{a}" link2="{b}" reason="Never"/>\n'
                        for a, b in itertools.combinations(links, 2))
        upstream = (CACHE / URDF.replace(".urdf", ".srdf")).read_text()
        srdf.write_text(upstream.replace("</robot>", extra + "</robot>"))
    return {"sim": sim, "plan": plan, "srdf": srdf}


def hand_qpos(arm: np.ndarray, fingers: float = 0.0, thumb: float = 0.0) -> np.ndarray:
    """Full joint vector (SAPIEN order) with q2 joints consistent with their q1."""
    q = np.zeros(17, dtype=np.float32)
    q[:7] = arm
    q[7] = -thumb
    q[8:12] = fingers
    q[13:17] = MIMIC_K * fingers + MIMIC_B
    return q


@register_agent()
class XArm7AbilityHand(XArm7Ability):
    uid = UID
    urdf_path = str(build_assets()["sim"])
    urdf_config = dict(
        _materials=dict(front_finger=dict(static_friction=2.0, dynamic_friction=1.5, restitution=0.0)),
        link={f"{n}_L2": dict(material="front_finger", patch_radius=0.05, min_patch_radius=0.04)
              for n in FINGERS},
    )

    @property
    def _controller_configs(self):
        self.arm_stiffness, self.arm_damping, self.arm_force_limit = 1e4, 1e3, 1000
        return super()._controller_configs

    def _after_init(self):
        super()._after_init()
        for link in self.robot.get_links():
            for body in link._bodies:
                body.disable_gravity = True
        self.pads = {l.name: l for l in self.robot.get_links() if l.name in {f"{n}_L2" for n in FINGERS}}

    def is_grasping(self, obj, min_force: float = 0.5, min_fingers: int = 2):
        """Held = at least two finger pads press on it (one of them the thumb's)."""
        import torch

        touching = []
        for name, link in self.pads.items():
            f = self.scene.get_pairwise_contact_forces(link, obj)
            touching.append((name, torch.linalg.norm(f, dim=-1) > min_force))
        count = sum(t.int() for _, t in touching)
        thumb = dict(touching)["thumb_L2"]
        return (count >= min_fingers) & thumb
