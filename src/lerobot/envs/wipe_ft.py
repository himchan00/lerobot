#!/usr/bin/env python
"""Wipe-FT: robosuite Wipe where wiping needs the right pressing force on a surface the cameras misjudge.

- The table's collision box is shifted from the drawn table by a hidden height offset (±15 mm) and tilt
  (±2° roll and pitch), so where the tool meets the table is not visible.
- A marker is wiped only while the tool covers it and presses with 5–20 N (robosuite counts any touch);
  over 35 N ends the episode as a failure. Success: all markers wiped.
- 40 markers along robosuite's random path (5 mm apart). Panda with robosuite's WipingGripper, OSC_POSE kp
  150, 20 Hz.

Same observations and wrapper as Square-FT (`square_ft.py`); the gripper has no fingers, so robot_state's
gripper entries are zeros and the 7-D action's last entry is ignored.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import mujoco
import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.wipe import DEFAULT_WIPE_CONFIG, Wipe
from robosuite.utils.mjcf_utils import array_to_string, string_to_array

from .square_ft import CAMERAS, SquareFTEnv, _SquareFTSyncVectorEnv
from .utils import _LazyAsyncVectorEnv

TASK = "wipe the marked path on the table"
NUM_MARKERS = 40
HEIGHT_RANGE = 0.015  # hidden collision-surface offset, ±
TILT_RANGE = np.deg2rad(2.0)  # hidden roll and pitch, ±
FORCE_WIPE = (5.0, 20.0)  # N, normal force that wipes
FORCE_FAIL = 35.0  # N


class WipeFTTask(Wipe):
    """Wipe with a hidden surface pose and force-gated wiping. `surface` = (dz, roll, pitch) of this episode."""

    def __init__(self, **kwargs):
        config = dict(DEFAULT_WIPE_CONFIG, num_markers=NUM_MARKERS, early_terminations=False)
        self.surface = np.zeros(3)
        self.overforce = False
        self.tool_force = 0.0
        super().__init__(task_config=config, **kwargs)

    def _load_model(self):
        super()._load_model()
        col = self.model.worldbody.find(".//geom[@name='table_collision']")
        dz = np.random.uniform(-HEIGHT_RANGE, HEIGHT_RANGE)
        roll, pitch = np.random.uniform(-TILT_RANGE, TILT_RANGE, size=2)
        self.surface = np.array([dz, roll, pitch])
        pos = string_to_array(col.get("pos"))
        col.set("pos", array_to_string(pos + [0, 0, dz]))  # in the xml: MuJoCo 3 fixes static geom poses
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, [roll, pitch, 0.0], "xyz")
        col.set("quat", array_to_string(quat))

    def _reset_internal(self):
        super()._reset_internal()
        self.overforce, self.tool_force = False, 0.0

    def _tool_table_force(self) -> float:
        """Normal force between the wiping tool and the table at the current state."""
        m, d = self.sim.model._model, self.sim.data._data
        table = self.sim.model.geom_name2id("table_collision")
        tool = {self.sim.model.geom_name2id(g) for g in self.robots[0].gripper.contact_geoms}
        f6, total = np.zeros(6), 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == table and c.geom2 in tool) or (c.geom2 == table and c.geom1 in tool):
                mujoco.mj_contactForce(m, d, i, f6)
                total += f6[0]
        return total

    def reward(self, action=None):
        """Wipe markers under the tool while it presses within FORCE_WIPE (replaces robosuite's rule)."""
        self.tool_force = self._tool_table_force()
        self.overforce |= self.tool_force > FORCE_FAIL
        if FORCE_WIPE[0] <= self.tool_force <= FORCE_WIPE[1]:
            corners = np.array(
                [self.sim.data.geom_xpos[self.sim.model.geom_name2id(g)][:2]
                 for g in self.robots[0].gripper.important_geoms["corners"]]
            )
            for marker in self.model.mujoco_arena.markers:
                if marker in self.wiped_markers:
                    continue
                p = self.sim.data.body_xpos[self.sim.model.body_name2id(marker.root_body)][:2]
                if _in_quad(p, corners):
                    self.sim.model.geom_rgba[self.sim.model.geom_name2id(marker.visual_geoms[0])][3] = 0
                    self.wiped_markers.append(marker)
        return len(self.wiped_markers) / self.num_markers

    def _check_success(self):
        return len(self.wiped_markers) == self.num_markers and not self.overforce


def _in_quad(p: np.ndarray, corners: np.ndarray) -> bool:
    """p inside the tool rectangle; corners in the gripper's order 1, 2, 3, 4 = (+x+y, -x+y, +x-y, -x-y)."""
    o, a, b = corners[1], corners[0] - corners[1], corners[3] - corners[1]
    u, v = (p - o) @ a / (a @ a), (p - o) @ b / (b @ b)
    return 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0


def make_wipe_ft_task(image_size: int | None = 256, kp_pos: float = 150.0, kp_rot: float = 150.0) -> WipeFTTask:
    controller = load_controller_config(default_controller="OSC_POSE")
    controller["kp"] = [kp_pos] * 3 + [kp_rot] * 3
    cams = image_size is not None
    return WipeFTTask(
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


class WipeFTEnv(SquareFTEnv):
    """SquareFTEnv on WipeFTTask: no fingers (zeros in robot_state), 7-D actions with the last entry ignored, and an
    episode also ends (as a failure) when the tool presses over FORCE_FAIL."""

    settle_action = np.zeros(6)

    def __init__(self, episode_length: int = 300, **kwargs):
        super().__init__(episode_length=episode_length, **kwargs)
        self.task, self.task_description = "wipe_ft", TASK

    def _ensure_env(self) -> None:
        if self._env is None:
            self._env = make_wipe_ft_task(self.image_size, kp_rot=self.kp_rot)

    def _format_raw_obs(self, raw_obs: dict):
        raw_obs = dict(raw_obs, robot0_gripper_qpos=np.zeros(2), robot0_gripper_qvel=np.zeros(2))
        return super()._format_raw_obs(raw_obs)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        raw_obs, reward, done, info = self._env.step(action[:6])
        is_success = bool(self._env._check_success())
        failed = self._env.overforce
        info.update({"task": self.task, "done": done, "is_success": is_success, "overforce": failed})
        observation = self._format_raw_obs(raw_obs)
        if is_success or failed:
            self.reset()
        return observation, reward, is_success or failed, False, info


def create_wipe_ft_envs(n_envs: int, gym_kwargs: dict[str, Any], env_cls) -> dict[str, dict[int, Any]]:
    """{"wipe_ft": {0: vec_env}}, as the single-task entry of `make_env`."""
    fns = [partial(WipeFTEnv, **gym_kwargs) for _ in range(n_envs)]
    if env_cls.__name__ != "AsyncVectorEnv":
        return {"wipe_ft": {0: _SquareFTSyncVectorEnv(fns)}}
    probe = WipeFTEnv(**gym_kwargs)
    return {"wipe_ft": {0: _LazyAsyncVectorEnv(fns, probe.observation_space, probe.action_space, probe.metadata)}}
