#!/usr/bin/env python
"""Wrist F/T observation for LIBERO (`observation.ft_wrench`, shape (N_SUBSTEPS, 6)).

Matches the regenerated dataset (`examples/port_datasets/libero_hf/regenerate_libero_hf.py`): at every
physics step of the last control step, the environment-interaction wrench at the wrist, i.e. the sum of
MuJoCo contact forces on the bodies below the PandaGripper F/T site (what a gravity- and inertia-
compensated wrist sensor reads), about the EE point (OSC `grip_site`) and in the EE frame, [F, tau] in
N and N m. The raw sensor also carries the gripper's inertial reaction to the last command, which a
policy learns to copy.

The logger leaves the rollout unchanged but refreshes derived quantities after every physics step, so
observables sample one physics step later than in the stock env (the state after substep 24 of 25
instead of 23, <0.5 mm). The regenerated dataset was logged the same way, so evaluate policies trained
on it with the logger on, with or without F/T input.
"""

from collections import deque
from typing import Any

import mujoco
import numpy as np

N_SUBSTEPS = 25  # 2 ms physics, 20 Hz control


class FTWrenchLogger:
    """Wraps `sim.step` of a robosuite env and keeps the contact wrench of the last N_SUBSTEPS physics steps.

    Each sample follows a forced controller update (`sim.forward()`), so contacts and the EE pose refer to
    the post-step state. The controller's update flag is restored, so the rollout is unchanged.
    """

    def __init__(self, robo_env: Any) -> None:
        robot = robo_env.robots[0]
        self.ctrl = robot.controller
        self.sim = robo_env.sim
        self.model, self.data = self.sim.model._model, self.sim.data._data
        ft_site = self.sim.model.site_name2id(f"{robot.gripper.naming_prefix}ft_frame")
        ft_body = int(self.model.site_bodyid[ft_site])
        self.below_ft = {b for b in range(self.model.nbody) if self._descends(b, ft_body)}
        n_sub = int(round(robo_env.control_timestep / robo_env.model_timestep))
        if n_sub != N_SUBSTEPS:
            raise ValueError(f"Expected {N_SUBSTEPS} physics steps per control step, got {n_sub}.")
        self._f6 = np.zeros(6)
        self.samples: deque[np.ndarray] = deque(maxlen=N_SUBSTEPS)
        self._orig_step = self.sim.step
        self.sim.step = self._step  # instance override; robosuite calls self.sim.step()

    def _descends(self, body: int, root: int) -> bool:
        while body not in (0, root):
            body = int(self.model.body_parentid[body])
        return body == root

    def detach(self) -> None:
        self.sim.step = self._orig_step

    def _step(self, *args: Any, **kwargs: Any) -> None:
        self._orig_step(*args, **kwargs)
        flag = self.ctrl.new_update
        self.ctrl.update(force=True)  # runs sim.forward()
        self.ctrl.new_update = flag
        self.samples.append(self._wrench())

    def _wrench(self) -> np.ndarray:
        p_ee = np.array(self.ctrl.ee_pos)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            in1 = int(self.model.geom_bodyid[c.geom1]) in self.below_ft
            in2 = int(self.model.geom_bodyid[c.geom2]) in self.below_ft
            if in1 == in2:  # internal to the gripper, or not touching it
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self._f6)
            frame = np.asarray(c.frame).reshape(3, 3)  # rows: normal (geom1 -> geom2), tangents
            sign = -1.0 if in1 else 1.0  # mj_contactForce gives the force on geom2
            force = sign * (frame.T @ self._f6[:3])
            wrench[:3] += force
            wrench[3:] += sign * (frame.T @ self._f6[3:]) + np.cross(np.asarray(c.pos) - p_ee, force)
        r_ee_t = np.array(self.ctrl.ee_ori_mat).T
        return np.concatenate([r_ee_t @ wrench[:3], r_ee_t @ wrench[3:]])

    def block(self) -> np.ndarray:
        if len(self.samples) != N_SUBSTEPS:
            raise RuntimeError(f"F/T logger holds {len(self.samples)} samples, expected {N_SUBSTEPS}.")
        return np.stack(self.samples).astype(np.float32)
