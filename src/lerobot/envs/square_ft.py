#!/usr/bin/env python
"""Square-FT: robosuite Square (NutAssemblySquare) where the final alignment needs the wrist F/T.

- The square peg is widened to a 2.25 mm clearance per side (peg 41 mm, nut hole 45.5 mm).
- Every episode its collision box is shifted from the drawn box by a hidden offset (L-inf 4-8 mm in the peg
  frame, always beyond the clearance). Cameras cannot locate the hole, so a nut aimed at the drawn peg lands
  on the peg top, and the way to the hole shows only in the contact wrench.
- Placement follows MimicGen Square D1: nut anywhere on the table, peg x in [-0.1, 0.3], y in [-0.2, 0.2].

Observations match the LIBERO wrapper (`libero.py`): raw renders of agentview and eye-in-hand, `robot_state`,
and `ft_wrench`, the EE-frame contact wrench of the last control step (`libero_ft.py`). Use LiberoProcessorStep.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.nut_assembly import NutAssemblySquare
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

from lerobot.types import RobotObservation

from .libero_ft import N_SUBSTEPS, FTWrenchLogger
from .utils import _LazyAsyncVectorEnv

TASK = "put the square nut on the square peg"
PEG_HALF = 0.0205  # nut hole half-width is 0.02275
OFFSET_RANGE = (0.004, 0.008)  # L-inf norm of the hidden collision-box offset
NUT_RANGE = ((-0.115, 0.115), (-0.255, 0.255))
PEG_RANGE = ((-0.1, 0.3), (-0.2, 0.2))
TABLE_TOP = np.array((0.0, 0.0, 0.82))
CAMERAS = ("agentview", "robot0_eye_in_hand")
SETTLE_STEPS = 10
SETTLE_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)


class SquareFTTask(NutAssemblySquare):
    """NutAssemblySquare with the widened, secretly offset peg. `peg_offset` holds this episode's offset."""

    def __init__(self, **kwargs):
        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for name, (x, y) in (("SquareNut", NUT_RANGE), ("RoundNut", ((-1.1, -1.0), (-1.1, -1.0)))):
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{name}Sampler",
                    x_range=x,
                    y_range=y,
                    rotation=(0.0, 2 * np.pi),
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=TABLE_TOP,
                    z_offset=0.02,
                )
            )
        self.peg_offset = np.zeros(2)
        super().__init__(placement_initializer=sampler, **kwargs)

    def _load_model(self):
        super()._load_model()
        peg1 = self.model.worldbody.find(".//body[@name='peg1']")
        pos = string_to_array(peg1.get("pos"))
        pos[:2] = TABLE_TOP[:2] + [np.random.uniform(*r) for r in PEG_RANGE]
        peg1.set("pos", array_to_string(pos))
        col, vis = peg1.findall("./geom")  # collision (group 0), drawn (group 1)
        for geom in (col, vis):
            size = string_to_array(geom.get("size"))
            size[:2] = PEG_HALF
            geom.set("size", array_to_string(size))
        col.set("name", "peg1_collision")
        col.set("friction", "0.3 0.005 0.0001")  # nut on peg: metal on metal (stock 1 and 0.95)
        # in the xml: MuJoCo 3 does not recompute static geom poses, so geom_pos edits after compile are ignored
        r = np.random.uniform(*OFFSET_RANGE)
        s = np.random.uniform(-r, r)
        self.peg_offset = np.array([(r, s), (-r, s), (s, r), (s, -r)][np.random.randint(4)])
        col.set("pos", array_to_string(np.r_[self.peg_offset, 0.0]))  # peg frame = world frame (no peg yaw)
        for geom in self.model.worldbody.findall(".//body[@name='SquareNut_main']//geom"):
            if geom.get("group") == "0":  # finger pads (friction 2) still grip the handle
                geom.set("friction", "0.3 0.3 0.1")
        peg2 = self.model.worldbody.find(".//body[@name='peg2']")
        peg2.set("pos", "-10 0 0.85")  # out of the scene, as in MimicGen Square

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return
        peg_xy = self.sim.data.body_xpos[self.peg1_body_id][:2]
        nut = self.nuts[self.nut_id]
        for _ in range(1000):  # keep the nut clear of the peg
            pos, quat, _ = self.placement_initializer.sample()[nut.name]
            if np.linalg.norm(np.asarray(pos[:2]) - peg_xy) > nut.horizontal_radius + PEG_HALF * np.sqrt(2):
                break
        self.sim.data.set_joint_qpos(nut.joints[0], np.concatenate([pos, quat]))


def make_square_ft_task(image_size: int | None = 256, kp_pos: float = 150.0, kp_rot: float = 150.0) -> SquareFTTask:
    """robosuite env on the OSC_POSE / 20 Hz setup of LIBERO and robomimic. `image_size=None` skips rendering."""
    controller = load_controller_config(default_controller="OSC_POSE")
    controller["kp"] = [kp_pos] * 3 + [kp_rot] * 3
    cams = image_size is not None
    return SquareFTTask(
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


class SquareFTEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    settle_action = SETTLE_ACTION

    def __init__(
        self,
        episode_length: int = 400,
        observation_height: int = 256,
        observation_width: int = 256,
        kp_rot: float = 150.0,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        if observation_height != observation_width:
            raise ValueError("Square-FT renders square images.")
        self.image_size = observation_height
        self.kp_rot = kp_rot
        self.render_mode = render_mode
        self.task = "square_ft"
        self.task_description = TASK
        self._max_episode_steps = episode_length
        self._env: SquareFTTask | None = None  # created in the worker on first reset (EGL context)
        self._ft_logger: FTWrenchLogger | None = None
        image = spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8)
        vec = lambda *shape: spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)  # noqa: E731
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict({"image": image, "image2": image}),
                "robot_state": spaces.Dict(
                    {
                        "eef": spaces.Dict({"pos": vec(3), "quat": vec(4), "mat": vec(3, 3)}),
                        "gripper": spaces.Dict({"qpos": vec(2), "qvel": vec(2)}),
                        "joints": spaces.Dict({"pos": vec(7), "vel": vec(7)}),
                    }
                ),
                "ft_wrench": spaces.Box(low=-np.inf, high=np.inf, shape=(N_SUBSTEPS, 6), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def _ensure_env(self) -> None:
        if self._env is None:
            self._env = make_square_ft_task(self.image_size, kp_rot=self.kp_rot)

    def _format_raw_obs(self, raw_obs: dict) -> RobotObservation:
        robot, data = self._env.robots[0], self._env.sim.data
        return {
            "pixels": {"image": raw_obs["agentview_image"], "image2": raw_obs["robot0_eye_in_hand_image"]},
            "robot_state": {
                "eef": {
                    "pos": raw_obs["robot0_eef_pos"],
                    "quat": raw_obs["robot0_eef_quat"],
                    "mat": robot.controller.ee_ori_mat,
                },
                "gripper": {"qpos": raw_obs["robot0_gripper_qpos"], "qvel": raw_obs["robot0_gripper_qvel"]},
                "joints": {  # not a default robosuite observable
                    "pos": np.array(data.qpos[robot._ref_joint_pos_indexes]),
                    "vel": np.array(data.qvel[robot._ref_joint_vel_indexes]),
                },
            },
            "ft_wrench": self._ft_logger.block(),
        }

    def render(self):
        self._ensure_env()
        return self._env._get_observations()["agentview_image"][::-1, ::-1]

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)  # robosuite placement uses the global numpy RNG
        self._env.reset()
        if self._ft_logger is not None:  # reset rebuilds the sim
            self._ft_logger.detach()
        self._ft_logger = FTWrenchLogger(self._env)
        for _ in range(SETTLE_STEPS):
            raw_obs, _, _, _ = self._env.step(self.settle_action)
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(f"Expected a 1-D action, got shape {action.shape}.")
        raw_obs, reward, done, info = self._env.step(action)
        is_success = bool(self._env._check_success())
        info.update({"task": self.task, "done": done, "is_success": is_success})
        observation = self._format_raw_obs(raw_obs)
        if is_success:
            self.reset()
        return observation, reward, is_success, False, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class _SquareFTSyncVectorEnv(gym.vector.SyncVectorEnv):
    """Allow repeated evaluation after close(), as `libero._LiberoSyncVectorEnv`."""

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.closed = False
        return super().reset(**kwargs)


def create_square_ft_envs(
    n_envs: int,
    gym_kwargs: dict[str, Any],
    env_cls: Callable[[list[Callable[[], Any]]], Any],
) -> dict[str, dict[int, Any]]:
    """{"square_ft": {0: vec_env}}, as the single-task entry of `make_env`."""
    fns = [partial(SquareFTEnv, **gym_kwargs) for _ in range(n_envs)]
    if env_cls is not gym.vector.AsyncVectorEnv:
        return {"square_ft": {0: _SquareFTSyncVectorEnv(fns)}}
    probe = SquareFTEnv(**gym_kwargs)
    return {"square_ft": {0: _LazyAsyncVectorEnv(fns, probe.observation_space, probe.action_space, probe.metadata)}}
