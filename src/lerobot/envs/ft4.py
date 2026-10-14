#!/usr/bin/env python
"""FT-4: robosuite Square, Stack, Wipe and Door under perception error.

The scenes are robosuite's own tasks. The only scene change is for the F/T: Square's square peg is 41 mm wide instead
of 32 mm, so the nut's 45.5 mm hole has 2.25 mm of clearance per side, and a nut placed further off must find the
hole by touch. Demonstrations come from ideal (non-reactive) experts, and from the same demonstrations converted to
perception error (examples/port_datasets/ft4). At evaluation the cameras' calibration is off by a random amount every
episode (`camera_error`, perturb_cameras), and Stack and Wipe apply contact rules: crushing or dropping the block,
or wiping outside 5-25 N, fails.

Panda, OSC_POSE kp 150, 20 Hz. Observations match the LIBERO wrapper (`libero.py`): raw renders of agentview and
eye-in-hand, `robot_state`, and `ft_wrench`, the EE-frame contact wrench of the last control step (`libero_ft.py`); use
LiberoProcessorStep. Wipe keeps robosuite's WipingGripper, which has no fingers: it steps with the first 6 action
entries, and its gripper state is zeros.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.nut_assembly import NutAssemblySquare
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.utils.mjcf_utils import array_to_string, string_to_array

from lerobot.lerobot_types import RobotObservation

from .libero_ft import N_SUBSTEPS, FTWrenchLogger
from .utils import _LazyAsyncVectorEnv

GRIP_OPEN = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
CAMERAS = ("agentview", "robot0_eye_in_hand")
SETTLE_STEPS = 10
CAMERA_ERROR = {"agentview": (0.005, 0.5), "robot0_eye_in_hand": (0.002, 0.5)}  # max shift (m) and turn (deg)


def perturb_cameras(sim, rng: np.random.Generator, scale: float = 1.0) -> None:
    """Calibration error (FT-4's perception error): shift each camera by up to its CAMERA_ERROR translation per axis
    and turn it about a random axis by up to its rotation, both times `scale`. The images change; the physics and
    the robot's state do not. A hard reset restores the cameras."""
    import mujoco

    for name, (shift, deg) in CAMERA_ERROR.items():
        i = sim.model.camera_name2id(name)
        sim.model.cam_pos[i] = sim.model.cam_pos[i] + rng.uniform(-shift, shift, 3) * scale
        axis = rng.normal(size=3)
        angle = np.deg2rad(rng.uniform(-deg, deg) * scale)
        turn, quat = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis / np.linalg.norm(axis)], np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, np.array(sim.model.cam_quat[i]))
        sim.model.cam_quat[i] = quat


def _net_force(sim, geoms_a: set[int], geoms_b: set[int]) -> np.ndarray:
    """World-frame net contact force exerted by geoms_b on geoms_a."""
    m, d = sim.model._model, sim.data._data
    f6, total = np.zeros(6), np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        if c.geom1 in geoms_a and c.geom2 in geoms_b:
            sign = -1.0  # mj_contactForce gives the force on geom2
        elif c.geom2 in geoms_a and c.geom1 in geoms_b:
            sign = 1.0
        else:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        total += sign * (np.asarray(c.frame).reshape(3, 3).T @ f6[:3])
    return total


def _geom_ids(sim, names) -> set[int]:
    return {sim.model.geom_name2id(n) for n in names}


# ----------------------------------------------------------------------------------------------------------------
# Square: the stock scene with a 41 mm square peg
# ----------------------------------------------------------------------------------------------------------------

PEG_HALF = 0.0205  # stock 0.016; the nut hole's half-width is 0.02275


class SquareFT4Task(NutAssemblySquare):
    """robosuite Square with the square peg (drawn and collision box) widened to 41 mm. `native` marks the scene of
    the non-reactive demonstrations (the experts read it); the scene is the same either way."""

    failed = False

    def __init__(self, native: bool = False, contact_rules: bool | None = None, **kwargs):
        self.native = native
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        for geom in self.model.worldbody.findall(".//body[@name='peg1']/geom"):
            size = string_to_array(geom.get("size"))
            size[:2] = PEG_HALF
            geom.set("size", array_to_string(size))


# ----------------------------------------------------------------------------------------------------------------
# Stack: set the red block down on the green one
# ----------------------------------------------------------------------------------------------------------------

STACK_FORCE_FAIL = 20.0  # N between the blocks: crushed
STACK_DROP_FAIL = 0.003  # m gap under the red block when the fingers let go: dropped


class StackFTTask(Stack):
    """robosuite Stack. With `contact_rules`, the red block must be released while touching (gap under 3 mm) and
    never pressed with more than 20 N. `native` marks the scene of the non-reactive demonstrations (the experts read
    it); the scene is the same either way."""

    def __init__(self, native: bool = False, contact_rules: bool | None = None, **kwargs):
        self.native = native
        self.contact_rules = not native if contact_rules is None else contact_rules
        super().__init__(**kwargs)

    def _setup_references(self):
        super()._setup_references()
        self.cubeA_geoms = _geom_ids(self.sim, self.cubeA.contact_geoms)
        self.cubeB_geoms = _geom_ids(self.sim, self.cubeB.contact_geoms)

    def _reset_internal(self):
        super()._reset_internal()
        self.overforce = self.dropped = False
        self.was_grasped, self.place_force = False, 0.0

    def true_top(self) -> float:
        return float(self.sim.data.body_xpos[self.cubeB_body_id][2] + 0.025)

    def reward(self, action=None):
        self.place_force = float(np.linalg.norm(_net_force(self.sim, self.cubeA_geoms, self.cubeB_geoms)))
        grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cubeA)
        if self.contact_rules:
            self.overforce |= self.place_force > STACK_FORCE_FAIL
            if self.was_grasped and not grasped and self.place_force == 0.0:
                gap = self.sim.data.body_xpos[self.cubeA_body_id][2] - 0.02 - self.true_top()
                self.dropped |= gap > STACK_DROP_FAIL
        self.was_grasped = grasped
        return super().reward(action)

    def _check_success(self):
        return super()._check_success() and not (self.overforce or self.dropped)

    @property
    def failed(self) -> bool:
        return self.overforce or self.dropped


# ----------------------------------------------------------------------------------------------------------------
# Wipe: robosuite's wiping pad on its 100-marker path
# ----------------------------------------------------------------------------------------------------------------

PAD_HALF = (0.06, 0.025)  # the WipingGripper's wiping face, 12 x 5 cm
WIPE_FORCE = (5.0, 25.0)  # N between pad and table that wipes
WIPE_FAIL = 40.0  # N
WIPE_EVEN = 0.6  # centre of pressure within this fraction of the pad's half sizes: even pressure


class WipeFT4Task(Wipe):
    """robosuite Wipe: robosuite counts a marker once the pad covers it within 2 cm, and its early terminations (the
    arm touching the table, joint limits) end the episode, here as a failure. With `contact_rules` a marker counts
    only while the pad presses the table with 5-25 N and evenly (centre of pressure in the middle 60 %), and over
    40 N fails."""

    def __init__(self, native: bool = False, contact_rules: bool | None = None, **kwargs):
        self.native = native
        self.contact_rules = not native if contact_rules is None else contact_rules
        super().__init__(**kwargs)

    def _setup_references(self):
        super()._setup_references()
        self.pad_geoms = _geom_ids(self.sim, self.robots[0].gripper.contact_geoms)
        self.table_geom = self.sim.model.geom_name2id("table_collision")
        self.corner_geoms = [self.sim.model.geom_name2id(g) for g in self.robots[0].gripper.important_geoms["corners"]]

    def _reset_internal(self):
        super()._reset_internal()
        self.overforce, self.tool_force, self.cop = False, 0.0, np.zeros(2)

    def tool_frame(self):
        """Centre of the pad's wiping face and its long / short / up axes (world)."""
        c1, c2, c3, _ = (np.array(self.sim.data.geom_xpos[g]) for g in self.corner_geoms)  # (+x+y, -x+y, +x-y, -x-y)
        long, short = (c1 - c2) / np.linalg.norm(c1 - c2), (c3 - c1) / np.linalg.norm(c3 - c1)
        return (c2 + c3) / 2, np.stack([long, short, np.cross(long, short)], axis=1)

    def _contacts(self):
        """Total normal force and centre of pressure (world) of the pad on the table."""
        m, d = self.sim.model._model, self.sim.data._data
        f6, total, moment = np.zeros(6), 0.0, np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} & self.pad_geoms and self.table_geom in (c.geom1, c.geom2):
                mujoco.mj_contactForce(m, d, i, f6)
                total += f6[0]
                moment += f6[0] * np.asarray(c.pos)
        return total, (moment / total if total > 0 else None)

    def reward(self, action=None):
        self.tool_force, cop = self._contacts()
        face, r = self.tool_frame()
        self.cop = np.zeros(2) if cop is None else np.array([(cop - face) @ r[:, 0], (cop - face) @ r[:, 1]])
        if self.contact_rules:
            self.overforce |= self.tool_force > WIPE_FAIL
            even = abs(self.cop[0]) < WIPE_EVEN * PAD_HALF[0] and abs(self.cop[1]) < WIPE_EVEN * PAD_HALF[1]
            if not (WIPE_FORCE[0] <= self.tool_force <= WIPE_FORCE[1] and even):
                return 0.0  # no marker counts at this step
        return super().reward(action)

    def _check_success(self):
        return super()._check_success() and not self.overforce

    @property
    def failed(self) -> bool:
        return self.overforce or self.check_contact(self.robots[0].robot_model) or self.robots[0].check_q_limits()


# ----------------------------------------------------------------------------------------------------------------
# Door: robosuite's latched door
# ----------------------------------------------------------------------------------------------------------------


class DoorFT4Task(Door):
    """robosuite Door, latched: the handle must be turned before the door opens. `native` marks the scene of the
    non-reactive demonstrations (the experts read it); the scene is the same either way."""

    failed = False

    def __init__(self, native: bool = False, contact_rules: bool | None = None, **kwargs):
        self.native = native
        super().__init__(**kwargs)


# ----------------------------------------------------------------------------------------------------------------
# factories, the gym wrapper and the suite
# ----------------------------------------------------------------------------------------------------------------

TASKS = {  # name: (class, instruction, settle action)
    "square_ft": (SquareFT4Task, "put the square nut on the square peg", GRIP_OPEN),
    "stack_ft": (StackFTTask, "stack the red block on the green block", GRIP_OPEN),
    "wipe_ft": (WipeFT4Task, "wipe the marked path on the table", np.zeros(6)),
    "door_ft": (DoorFT4Task, "open the door", GRIP_OPEN),
}
FT4_TASKS = tuple(TASKS)
EPISODE_LENGTH = {"square_ft": 400, "stack_ft": 300, "wipe_ft": 600, "door_ft": 350}


def make_ft4_task(name: str, image_size: int | None = 256, native: bool = False, camera_depths: bool = False,
                  contact_rules: bool | None = None):
    """robosuite env of an FT-4 task: Panda, OSC_POSE kp 150, 20 Hz. `image_size=None` skips rendering."""
    controller = load_controller_config(default_controller="OSC_POSE")
    controller["kp"] = [150.0] * 6
    cams = image_size is not None
    return TASKS[name][0](
        native=native,
        contact_rules=contact_rules,
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=cams,
        use_camera_obs=cams,
        camera_names=list(CAMERAS),
        camera_heights=image_size or 84,
        camera_widths=image_size or 84,
        camera_depths=camera_depths,
        control_freq=20,
        ignore_done=True,
        reward_shaping=False,
    )


class FT4Env(gym.Env):
    """An FT-4 task as a gym env. An episode ends at success, or as a failure when a contact rule is broken or Wipe
    terminates early."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task: str, contact_rules: bool = True, episode_length: int | None = None,
                 observation_height: int = 256, observation_width: int = 256, render_mode: str = "rgb_array",
                 camera_error: float = 0.0):
        super().__init__()
        if observation_height != observation_width:
            raise ValueError("FT-4 renders square images.")
        self.image_size = observation_height
        self.render_mode = render_mode
        self.camera_error = camera_error  # scale of the per-episode camera calibration error (perturb_cameras)
        self.task, self.task_description, self.settle_action = task, TASKS[task][1], TASKS[task][2]
        self.contact_rules = contact_rules
        self._max_episode_steps = episode_length or EPISODE_LENGTH[task]
        self._env = None  # created in the worker on first reset (EGL context)
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
            self._env = make_ft4_task(self.task, self.image_size, native=True, contact_rules=self.contact_rules)

    def _format_raw_obs(self, raw_obs: dict) -> RobotObservation:
        robot, data = self._env.robots[0], self._env.sim.data
        if not robot.gripper.dof:  # the wiping pad has no fingers
            raw_obs = dict(raw_obs, robot0_gripper_qpos=np.zeros(2), robot0_gripper_qvel=np.zeros(2))
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
        if self.camera_error:
            perturb_cameras(self._env.sim, self.np_random, self.camera_error)
        if self._ft_logger is not None:  # reset rebuilds the sim
            self._ft_logger.detach()
        self._ft_logger = FTWrenchLogger(self._env)
        for _ in range(SETTLE_STEPS):
            raw_obs, _, _, _ = self._env.step(self.settle_action)
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(f"Expected a 1-D action, got shape {action.shape}.")
        raw_obs, reward, done, info = self._env.step(action[: self._env.action_dim])
        is_success = bool(self._env._check_success())
        failed = bool(self._env.failed)
        info.update({"task": self.task, "done": done, "is_success": is_success, "failed": failed})
        observation = self._format_raw_obs(raw_obs)
        if is_success or failed:
            self.reset()
        return observation, reward, is_success or failed, False, info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class _FT4SyncVectorEnv(gym.vector.SyncVectorEnv):
    """Allow repeated evaluation after close() (eval closes each task env after use)."""

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.closed = False
        return super().reset(**kwargs)


def create_ft4_envs(n_envs: int, gym_kwargs: dict[str, Any], env_cls, tasks=(), camera_error: float = 1.0,
                    contact_rules: bool = True) -> dict[str, dict[int, Any]]:
    """{task: {0: vec_env}} for the FT-4 suite (or the given `tasks`), each with its own episode length."""
    out = {}
    for t in tasks or FT4_TASKS:
        kwargs = {**gym_kwargs, "episode_length": EPISODE_LENGTH[t], "camera_error": camera_error}
        make = partial(FT4Env, t, contact_rules, **kwargs)
        fns = [make for _ in range(n_envs)]
        if env_cls.__name__ != "AsyncVectorEnv":
            out[t] = {0: _FT4SyncVectorEnv(fns)}
        else:
            probe = make()
            out[t] = {0: _LazyAsyncVectorEnv(fns, probe.observation_space, probe.action_space, probe.metadata)}
    return out
