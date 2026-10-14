#!/usr/bin/env python
"""FT-4's Square expert (lerobot/envs/ft4.py: robosuite Square with a 41 mm square peg).

It grasps the nut by its handle, carries it over the peg turned so that the handle faces the robot, and lowers it.
Native (`env.native`): it aims at the peg and lowers the nut straight onto it. Reactive (convert_native.py, aim off by
the perception error): when the nut lands on the peg top it presses for two steps, reads the support's centre of
pressure (CoP) from the wrist wrench, then slides the nut toward the CoP while still touching the peg top, re-reading
it every step, until the nut drops into the hole.
"""

import numpy as np

HOLE_HALF = 0.02275  # square nut hole half-width
NUT_HALF_THICK = 0.01
WINDOW = 10  # physics samples (of 25) averaged for the wrench
CONTACT_N = 2.5  # upward force on the gripper that counts as landing on the peg top
SLIDE_LEAD = 0.006  # nut target ahead of the nut along the CoP direction
SLIDE_PUSH = 0.003  # nut target below the landing height (light preload that keeps the CoP readable)
DROP = 0.004  # the nut is this far below its landing height: it is in the hole
SUPPORT_LOST = 0.8  # upward force (N) below which the nut no longer rests on the peg top
CARRY = 0.04  # nut bottom above the peg top while carried (the nut hangs tilted from its handle)

APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, PROBE, PRESS, SLIDE, INSERT = range(9)
N_PHASES = 9


def axisangle(mat: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(mat) - 1) / 2, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([mat[2, 1] - mat[1, 2], mat[0, 2] - mat[2, 0], mat[1, 0] - mat[0, 1]])
    return axis / (2 * np.sin(angle)) * angle


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def level(z: np.ndarray) -> np.ndarray:
    """Rotation that turns the unit vector z onto world z."""
    axis = np.cross(z, [0, 0, 1.0])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = np.arctan2(s, z[2])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx


class Expert:
    """Phases: approach, descend, grasp, lift, transport, probe, (press, slide)*, insert."""

    def __init__(self, env, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.ctrl = env.robots[0].controller
        sim = env.sim
        self.nut_id = env.obj_body_id[env.nuts[env.nut_id].name]
        self.handle_site = sim.model.site_name2id("SquareNut_handle_site")
        peg = np.array(sim.data.body_xpos[env.peg1_body_id])
        self.peg_top = peg[2] + 0.1
        self.aim = peg[:2].copy()  # where the expert believes the hole is (convert_native.py shifts it)
        left, right = (
            np.array(sim.data.body_xpos[sim.model.body_name2id(f"gripper0_{n}finger")]) for n in ("left", "right")
        )
        v = np.asarray(self.ctrl.ee_ori_mat).T @ (left - right)
        self.finger_axis = int(np.argmax(np.abs(v[:2])))  # finger closing axis in the EE frame
        self.grasp_shift = rng.uniform(-0.005, 0.005)  # along the handle
        self.phase, self.t_phase = APPROACH, 0
        self.turn = None  # sense (about world z) in which the nut is turned while carried; chosen in TRANSPORT
        self.f_sign = None  # which of the two finger directions across the handle the grasp takes; chosen once
        self.tare, self.contact_z, self.dir, self.presses, self.misses = np.zeros(6), None, None, 0, 0
        self.failed = False  # gave up (grasp lost, jammed); such episodes are dropped
        self.native = getattr(env, "native", False)  # native Square: lower the nut straight onto the peg

    def _ee(self):
        return np.array(self.ctrl.ee_pos), np.array(self.ctrl.ee_ori_mat)

    def _nut(self):
        d = self.env.sim.data
        return np.array(d.body_xpos[self.nut_id]), np.array(d.body_xmat[self.nut_id]).reshape(3, 3)

    def _world_wrench(self, block: np.ndarray, window: int = WINDOW) -> np.ndarray:
        w = block[-window:].mean(0)
        r = np.array(self.ctrl.ee_ori_mat)
        return np.concatenate([r @ w[:3], r @ w[3:]])

    def _grasp_rot(self) -> np.ndarray:
        _, r_nut = self._nut()
        h = r_nut[:, 0].copy()
        h[2] = 0
        h /= np.linalg.norm(h)
        f = np.cross([0, 0, 1.0], h)  # across the handle
        _, r_ee = self._ee()
        if self.f_sign is None:
            self.f_sign = self._finger_sense(f, r_ee)
        f = self.f_sign * f
        z = np.array([0, 0, -1.0])
        cols = [None, None, z]
        cols[self.finger_axis] = f
        cols[1 - self.finger_axis] = np.cross(f, z) if self.finger_axis == 1 else np.cross(z, f)
        return np.stack(cols, axis=1)

    def _handle_point(self) -> np.ndarray:
        _, r_nut = self._nut()
        return np.array(self.env.sim.data.site_xpos[self.handle_site]) + r_nut[:, 0] * self.grasp_shift

    def _nut_target(self, nut_xy: np.ndarray, nut_z: float):
        """EE pose that puts the nut center at (nut_xy, nut_z) with the handle toward the robot (-x; the hole is then
        square to the peg); within 1 cm of the target, also with the nut level (it hangs tilted from its handle;
        leveling while carrying swings it). The handle toward the robot keeps the hand on the near side of the peg:
        the native expert failed with the handle away (out of reach) or sideways (jammed)."""
        p_ee, r_ee = self._ee()
        p_nut, r_nut = self._nut()
        dyaw = self._dyaw()
        rz = rot_z(np.clip(dyaw, -np.pi / 2, np.pi / 2))  # within a quarter turn: the tracker takes the shorter sense
        if np.linalg.norm(nut_xy - p_nut[:2]) < 0.01:
            rz = rz @ level(r_nut[:, 2])
        rel = rz @ (p_ee - p_nut)
        return np.array([nut_xy[0] + rel[0], nut_xy[1] + rel[1], nut_z + rel[2]]), rz @ r_ee, abs(dyaw)

    def _dyaw(self) -> float:
        """Turn about world z that brings the handle (nut x) to -x, in the sense self.turn once that is set."""
        _, r_nut = self._nut()
        dyaw = (-np.arctan2(r_nut[1, 0], r_nut[0, 0])) % (2 * np.pi) - np.pi
        if self.turn is not None and abs(dyaw) > np.pi / 2 and np.sign(dyaw) != self.turn:
            dyaw += 2 * np.pi * self.turn
        return dyaw

    def _wrist(self):
        """Last wrist joint: angle, range, and a (the joint turns the hand by a * dq about world z; a < 0)."""
        sim = self.env.sim
        j = sim.model.joint_name2id(self.env.robots[0].robot_joints[-1])
        return sim.data.qpos[sim.model.jnt_qposadr[j]], sim.model.jnt_range[j], sim.data.xaxis[j][2]

    def _finger_sense(self, f: np.ndarray, r_ee: np.ndarray) -> float:
        """The finger direction (+-f) closest to the hand's, unless the turn onto it or the shorter turn of the nut
        afterwards takes the last wrist joint within 0.3 rad of its range; then the other one. Turned the long way,
        the nut took the elbow to its limit, and the hand no longer held its pose."""
        q, (lo, hi), a = self._wrist()
        g = r_ee[:, self.finger_axis]
        turn = self._dyaw()
        for s in (1.0, -1.0) if f @ g >= 0 else (-1.0, 1.0):
            d = np.arctan2(g[0] * s * f[1] - g[1] * s * f[0], g @ (s * f))  # turn of the hand onto s * f
            if all(lo + 0.3 < x < hi - 0.3 for x in (q + d / a, q + (d + turn) / a)):
                return s
        return 1.0 if f @ g >= 0 else -1.0

    def _turn_sense(self) -> float:
        """The shorter sense, unless it takes the last wrist joint within 0.3 rad of its range (the long way then)."""
        dyaw = self._dyaw()
        q, (lo, hi), a = self._wrist()
        s = np.sign(dyaw) or 1.0
        for d in (dyaw, dyaw - 2 * np.pi * s):
            if lo + 0.3 < q + d / a < hi - 0.3:
                return float(np.sign(d))
        return float(s)

    @staticmethod
    def _track(p_ee, r_ee, p_tgt, r_tgt, max_step, max_rot=0.15):
        dp = p_tgt - p_ee
        n = np.linalg.norm(dp)
        if n > max_step:
            dp *= max_step / n
        rv = axisangle(r_tgt @ r_ee.T)
        n = np.linalg.norm(rv)
        if n > max_rot:
            rv *= max_rot / n
        return np.concatenate([np.clip(dp / 0.05, -1, 1), np.clip(rv / 0.5, -1, 1)])

    def _cop(self, dw, p_ee, p_nut):
        """Support center of pressure relative to the hole center, or None if it is not on the ring next to
        the hole. dw: wrench change since free space; rz corrects the torque of horizontal (friction) forces."""
        if dw[2] < 1.0:
            return None
        rz = self.peg_top - p_ee[2]
        c = p_ee[:2] + np.array([rz * dw[0] - dw[4], dw[3] + rz * dw[1]]) / dw[2] - p_nut[:2]
        return c if 0.018 < np.abs(c).max() < 0.032 else None

    def _go(self, phase):
        self.phase, self.t_phase = phase, 0

    def _press(self, p_ee, r_ee):
        self.press, self.press_tgt, self.misses = [], (np.r_[p_ee[:2], p_ee[2] - 0.006], r_ee), 0
        self._go(PRESS)
        return self._track(p_ee, r_ee, *self.press_tgt, 0.01)

    def act(self, block: np.ndarray) -> np.ndarray:
        """Next action from the sim state and the wrench block (25, 6) of the last control step."""
        p_ee, r_ee = self._ee()
        p_nut, _ = self._nut()
        grip, noise = 1.0, 0.02
        self.t_phase += 1
        if self.phase == APPROACH:
            grip, r_tgt = -1.0, self._grasp_rot()
            p_tgt = self._handle_point() + [0, 0, 0.06]
            a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.03)
            aligned = np.linalg.norm(axisangle(r_tgt @ r_ee.T)) < 0.05
            if np.linalg.norm(p_tgt - p_ee) < 0.01 and (aligned or self.t_phase > 150):  # a half turn: ~100 steps
                self._go(DESCEND)
        elif self.phase == DESCEND:
            grip, r_tgt = -1.0, self._grasp_rot()
            p_tgt = self._handle_point() + [0, 0, 0.003]
            a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.01)
            err = p_tgt - p_ee
            if np.linalg.norm(err[:2]) < 0.004 and (abs(err[2]) < 0.004 or self.t_phase > 30):  # fingers on table
                self._go(GRASP)
        elif self.phase == GRASP:
            a = self._track(p_ee, r_ee, p_ee, r_ee, 0.01)
            if self.t_phase >= 10:
                self._go(LIFT)
        elif self.phase == LIFT:
            p_tgt, _, _ = self._nut_target(p_nut[:2], self.peg_top + NUT_HALF_THICK + CARRY + 0.015)
            p_tgt[:2] = p_ee[:2]
            a = self._track(p_ee, r_ee, p_tgt, r_ee, 0.03)
            if p_nut[2] > self.peg_top + NUT_HALF_THICK + CARRY:
                self._go(TRANSPORT)
            self.failed = self.t_phase > 60  # grasp missed
        elif self.phase == TRANSPORT:
            if self.turn is None:
                self.turn = self._turn_sense()
            z = self.peg_top + NUT_HALF_THICK + CARRY
            p_tgt, r_tgt, dyaw = self._nut_target(self.aim, z)
            self.failed = self.t_phase > 250  # a half turn of the nut takes about 140 steps
            near = np.linalg.norm(p_nut[:2] - self.aim) < 0.01
            noise = 0.01 if near else noise
            a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.01 if near else 0.03)
            if np.linalg.norm(p_nut[:2] - self.aim) < 0.002 and dyaw < 0.02 and abs(p_nut[2] - z) < 0.005:
                self._go(INSERT if self.native else PROBE)
                self.probe_z = p_nut[2]
        elif self.phase == PROBE:
            noise = 0.01
            w = self._world_wrench(block)
            if self.t_phase == 1:
                self.tare = w  # free space: nut weight and inertia
            if p_nut[2] < self.peg_top + NUT_HALF_THICK - 0.008:  # slid into the hole
                self._go(INSERT)
                a = self._track(p_ee, r_ee, p_ee, r_ee, 0.01)
            elif w[2] - self.tare[2] > CONTACT_N:  # landed on the peg top
                self.contact_z = p_nut[2]
                a = self._press(p_ee, r_ee)
            else:
                self.probe_z -= 0.003
                p_tgt, r_tgt, _ = self._nut_target(self.aim, self.probe_z)
                a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.01)
        elif self.phase == PRESS:  # steady wrench for the CoP
            noise = 0.0
            if self.t_phase > 1:  # skip the landing transient
                self.press.append(self._world_wrench(block, window=25))
            if p_nut[2] < self.contact_z - DROP:
                self._go(INSERT)
            elif self.t_phase >= 3:
                c = self._cop(np.mean(self.press, axis=0) - self.tare, p_ee, p_nut)
                if c is not None:
                    self.dir = c / np.linalg.norm(c)
                elif self.dir is None and self.t_phase < 6:  # unclear: press harder
                    self.press_tgt[0][2] -= 0.002
                elif self.dir is None:  # still unclear: try a random way
                    angle = self.rng.uniform(0, 2 * np.pi)
                    self.dir = np.array([np.cos(angle), np.sin(angle)])
                if self.dir is not None:  # else keep pressing; a missed reading keeps the last direction
                    self.slide_from = p_nut[:2].copy()
                    self._go(SLIDE)
            a = self._track(p_ee, r_ee, *self.press_tgt, 0.01)
        elif self.phase == SLIDE:  # keep touching the peg top; every step, slide toward the current CoP
            noise = 0.005
            dw = self._world_wrench(block, window=25) - self.tare
            if p_nut[2] < self.contact_z - DROP or (dw[2] < SUPPORT_LOST and self.t_phase > 2):  # over the hole
                self._go(INSERT)
                a = self._track(p_ee, r_ee, p_ee, r_ee, 0.01)
            else:
                c = self._cop(dw, p_ee, p_nut)
                self.misses = 0 if c is not None else self.misses + 1
                if c is not None:
                    self.dir = c / np.linalg.norm(c)
                if self.misses >= 2:  # lost the reading: press and read again
                    self.presses += 1
                    self.failed = self.presses > 6
                    a = self._press(p_ee, r_ee)
                else:
                    lead = p_nut[:2] + SLIDE_LEAD * self.dir
                    p_tgt, r_tgt, _ = self._nut_target(lead, self.contact_z - SLIDE_PUSH)
                    a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.01)
                    self.failed = self.t_phase > 40
        elif self.phase == INSERT:
            noise = 0.01
            dw = self._world_wrench(block, window=25) - self.tare
            if not self.native and self.t_phase <= 6 and p_nut[2] > self.peg_top and dw[2] > CONTACT_N:  # caught on the rim
                self.contact_z = p_nut[2]
                a = self._press(p_ee, r_ee)
            elif self.native and p_nut[2] - NUT_HALF_THICK > self.peg_top:  # native: 3 mm down per step, on the peg
                p_tgt, r_tgt, _ = self._nut_target(self.aim, p_nut[2] - 0.003)
                a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.01)
            else:  # gently first, while the nut may still sit on the rim
                p_tgt, r_tgt, _ = self._nut_target(p_nut[:2], self.env.table_offset[2] + 0.012)
                a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.003 if self.t_phase <= 6 else 0.01)
                self.failed = self.t_phase > (150 if self.native else 80)  # jammed
        a = a + noise * self.rng.standard_normal(6)
        return np.concatenate([np.clip(a, -1, 1), [grip]])
