#!/usr/bin/env python
"""Door-FT: robosuite Door where the cameras cannot tell which side the door swings on.

- The handle is a bar centered on the panel and both frame posts look alike, but the hinge sits on the left or
  the right post at random (hidden). Pulling the handle shows the hinge side first in the wrist wrench (the door
  pulls the hand along its arc).
- Hinge range 0–1.6 rad; success when the door is open past 0.7 rad. Pulling against the hinge (net hand–door
  force over FORCE_FAIL) ends the episode as a failure.
- Door without the latch; Panda, OSC_POSE kp 150, 20 Hz. Same observations and wrapper as Square-FT.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import mujoco
import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.door import Door

from .square_ft import CAMERAS, SquareFTEnv, _SquareFTSyncVectorEnv
from .utils import _LazyAsyncVectorEnv

TASK = "open the door"
OPEN_RAD = 0.7  # the arm reaches its limits near 0.8 rad on the left hinge
FORCE_FAIL = 100.0  # N (pulling straight to OPEN_RAD would need ~110 N)


class DoorFTTask(Door):
    """Door with a centered handle and a hidden hinge side. `hinge_side` = +1 (right post) or -1 (left post)."""

    def __init__(self, **kwargs):
        self.hinge_side = 1
        self.overforce = False
        self.hand_force = 0.0
        super().__init__(use_latch=False, **kwargs)

    def _load_model(self):
        super()._load_model()
        wb = self.model.worldbody
        self.hinge_side = int(np.random.choice([-1, 1]))
        hinge = wb.find(".//joint[@name='Door_hinge']")
        hinge.set("pos", f"{0.255 * self.hinge_side} 0 0")
        hinge.set("axis", f"0 0 {self.hinge_side}")  # opens toward the robot for both sides
        hinge.set("range", "0 1.6")
        wb.find(".//body[@name='Door_latch']").set("pos", "0 0 -0.025")  # handle at the panel center
        for geom in wb.findall(".//body[@name='Door_latch']/geom"):
            if geom.get("name", "").startswith("Door_handle") and not geom.get("name", "").startswith("Door_handle_base"):
                geom.set("pos", "0 -0.10 0")  # bar centered on its base: symmetric
        wb.find(".//site[@name='Door_handle']").set("pos", "0 -0.10 0")

    def _reset_internal(self):
        super()._reset_internal()
        self.overforce, self.hand_force = False, 0.0

    def _hand_door_force(self) -> float:
        """|net force| of the door on the gripper (internal grip squeeze cancels)."""
        m, d = self.sim.model._model, self.sim.data._data
        door = {self.sim.model.body_name2id(b) for b in (self.door.door_body, self.door.latch_body)}
        hand = {self.sim.model.geom_name2id(g) for g in self.robots[0].gripper.contact_geoms}
        f6, total = np.zeros(6), np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in hand and m.geom_bodyid[g2] in door or g2 in hand and m.geom_bodyid[g1] in door:
                mujoco.mj_contactForce(m, d, i, f6)
                force = np.asarray(c.frame).reshape(3, 3).T @ f6[:3]
                total += -force if g1 in hand else force
        return float(np.linalg.norm(total))

    def reward(self, action=None):
        self.hand_force = self._hand_door_force()
        self.overforce |= self.hand_force > FORCE_FAIL
        return float(self.sim.data.qpos[self.hinge_qpos_addr])

    def _check_success(self):
        return self.sim.data.qpos[self.hinge_qpos_addr] > OPEN_RAD and not self.overforce


def make_door_ft_task(image_size: int | None = 256, kp_pos: float = 150.0, kp_rot: float = 150.0) -> DoorFTTask:
    controller = load_controller_config(default_controller="OSC_POSE")
    controller["kp"] = [kp_pos] * 3 + [kp_rot] * 3
    cams = image_size is not None
    return DoorFTTask(
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=cams,
        use_camera_obs=cams,
        camera_names=list(CAMERAS),
        camera_heights=image_size or 84,
        camera_widths=image_size or 84,
        control_freq=20,
        ignore_done=True,
        reward_shaping=False,
    )


class DoorFTEnv(SquareFTEnv):
    """SquareFTEnv on DoorFTTask; an episode also ends (as a failure) when the hand pulls over FORCE_FAIL."""

    def __init__(self, episode_length: int = 300, **kwargs):
        super().__init__(episode_length=episode_length, **kwargs)
        self.task, self.task_description = "door_ft", TASK

    def _ensure_env(self) -> None:
        if self._env is None:
            self._env = make_door_ft_task(self.image_size, kp_rot=self.kp_rot)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raw_obs, reward, done, info = self._env.step(action)
        is_success = bool(self._env._check_success())
        failed = self._env.overforce
        info.update({"task": self.task, "done": done, "is_success": is_success, "overforce": failed})
        observation = self._format_raw_obs(raw_obs)
        if is_success or failed:
            self.reset()
        return observation, reward, is_success or failed, False, info


def create_door_ft_envs(n_envs: int, gym_kwargs: dict[str, Any], env_cls) -> dict[str, dict[int, Any]]:
    """{"door_ft": {0: vec_env}}, as the single-task entry of `make_env`."""
    fns = [partial(DoorFTEnv, **gym_kwargs) for _ in range(n_envs)]
    if env_cls.__name__ != "AsyncVectorEnv":
        return {"door_ft": {0: _SquareFTSyncVectorEnv(fns)}}
    probe = DoorFTEnv(**gym_kwargs)
    return {"door_ft": {0: _LazyAsyncVectorEnv(fns, probe.observation_space, probe.action_space, probe.metadata)}}
