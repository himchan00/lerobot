#!/usr/bin/env python
"""Scripted experts for FT-4 (lerobot/envs/ft4.py): Stack, Wipe and the latched Door; Square's is square_expert.py.

Every expert plans from the scene (object poses, drawn geometry). In contact it reacts to the wrist wrench block
(25 physics samples, EE frame) of the last control step. On a native task (`env.native`) the expert skips the force
reaction and executes its plan for the drawn scene: the ideal demonstrations. convert_native.py runs the reactive
part with a wrong estimate of the scene (perception error).
"""

from __future__ import annotations

import numpy as np

import square_expert as sq

axisangle, track = sq.axisangle, sq.Expert._track
DOWN = np.array([0.0, 0.0, -1.0])

def finger_axis(env) -> int:
    """EE-frame axis (0 or 1) along which the fingers close."""
    sim, ctrl = env.sim, env.robots[0].controller
    left, right = (np.array(sim.data.body_xpos[sim.model.body_name2id(f"gripper0_{n}finger")]) for n in ("left", "right"))
    return int(np.argmax(np.abs((np.asarray(ctrl.ee_ori_mat).T @ (left - right))[:2])))

def top_down(f: np.ndarray, r_ee: np.ndarray, axis: int) -> np.ndarray:
    """Gripper pointing down, fingers closing along the horizontal direction f (the sign nearer to now)."""
    f = np.array([f[0], f[1], 0.0])
    f /= np.linalg.norm(f)
    if f @ r_ee[:, axis] < 0:
        f = -f
    cols = [None, None, DOWN]
    cols[axis] = f
    cols[1 - axis] = np.cross(f, DOWN) if axis == 1 else np.cross(DOWN, f)
    return np.stack(cols, axis=1)

def world_wrench(r_ee: np.ndarray, block: np.ndarray, window: int = 10) -> np.ndarray:
    w = block[-window:].mean(0)
    return np.concatenate([r_ee @ w[:3], r_ee @ w[3:]])

class Base:
    def __init__(self, env, rng):
        self.env, self.rng = env, rng
        self.ctrl = env.robots[0].controller
        self.axis = finger_axis(env) if env.robots[0].gripper.dof else None  # the wiping pad has no fingers
        self.native = env.native
        self.phase, self.t_phase, self.failed = 0, 0, False

    def ee(self):
        return np.array(self.ctrl.ee_pos), np.array(self.ctrl.ee_ori_mat)

    def go(self, phase):
        self.phase, self.t_phase = phase, 0

    def body(self, body_id):
        d = self.env.sim.data
        return np.array(d.body_xpos[body_id]), np.array(d.body_xmat[body_id]).reshape(3, 3)

    def finish(self, a, grip, noise):
        a = a + noise * self.rng.standard_normal(6)
        return np.concatenate([np.clip(a, -1, 1), [grip]])

class StackExpert(Base):
    """Grasp the red block, carry it over the green one, lower it slowly until the wrist feels the support, let go.
    Native: let go 1 mm above the drawn top."""

    APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, LOWER, PROBE, RELEASE, RETREAT = range(9)
    CONTACT_N = 1.5  # support force over the free-space tare that counts as touching

    def act(self, block):
        env = self.env
        p, r = self.ee()
        pa, ra = self.body(env.cubeA_body_id)
        pb, _ = self.body(env.cubeB_body_id)
        top = pb[2] + 0.025  # drawn top of the green block
        grip, noise = 1.0, 0.01
        self.t_phase += 1
        if self.phase == self.APPROACH:
            grip = -1.0
            f = max((ra[:, 0], ra[:, 1]), key=lambda v: abs(v[:2] @ r[:2, self.axis]))
            self.r_grasp = top_down(f, r, self.axis)
            tgt = pa + [0, 0, 0.06]
            a = track(p, r, tgt, self.r_grasp, 0.03)
            if np.linalg.norm(tgt - p) < 0.01 and np.linalg.norm(axisangle(self.r_grasp @ r.T)) < 0.05:
                self.go(self.DESCEND)
            self.failed = self.t_phase > 80
        elif self.phase == self.DESCEND:
            grip = -1.0
            a = track(p, r, pa, self.r_grasp, 0.01)
            if np.linalg.norm(pa - p) < 0.004 or self.t_phase > 30:
                self.go(self.GRASP)
        elif self.phase == self.GRASP:
            a = track(p, r, p, r, 0.01)
            if self.t_phase >= 10:
                self.go(self.LIFT)
        elif self.phase == self.LIFT:
            z = top + 0.02 + 0.05
            a = track(p, r, np.r_[p[:2], z], r, 0.03)
            if pa[2] > z - 0.01:
                self.go(self.TRANSPORT)
            self.failed = self.t_phase > 60 or pa[2] < top - 0.04 and self.t_phase > 20  # lost the block
        elif self.phase == self.TRANSPORT:
            z = top + 0.02 + 0.03
            tgt = p + np.r_[pb[:2] - pa[:2], z - pa[2]]
            a = track(p, r, tgt, r, 0.02)
            if np.linalg.norm(pb[:2] - pa[:2]) < 0.003 and abs(pa[2] - z) < 0.005:
                self.go(self.LOWER)
            self.failed = self.t_phase > 100
        elif self.phase == self.LOWER:  # fast to just above any possible support height
            z = top + 0.02 + (0.001 if self.native else 0.015)
            tgt = p + np.r_[pb[:2] - pa[:2], z - pa[2]]
            a = track(p, r, tgt, r, 0.01)
            if abs(pa[2] - z) < 0.002:
                self.tare, self.z_tgt = world_wrench(r, block), p[2]
                self.go(self.RELEASE if self.native else self.PROBE)
        elif self.phase == self.PROBE:  # 2 mm per step until the support pushes back
            noise = 0.005
            if world_wrench(r, block)[2] - self.tare[2] > self.CONTACT_N:
                self.z_tgt = p[2]
                self.go(self.RELEASE)
            else:
                self.z_tgt -= 0.002
            a = track(p, r, np.r_[p[:2] + pb[:2] - pa[:2], self.z_tgt], r, 0.01)
            self.failed = self.t_phase > 40
        elif self.phase == self.RELEASE:
            grip, noise = -1.0, 0.0
            a = track(p, r, np.r_[p[:2], self.z_tgt], r, 0.005)
            if self.t_phase >= 8:
                self.go(self.RETREAT)
        else:
            grip = -1.0
            a = track(p, r, np.r_[p[:2], top + 0.12], r, 0.02)
            self.failed = self.t_phase > 30
        return self.finish(a, grip, noise)

def rot_about(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector w."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx

def tilt_to_level(up: np.ndarray) -> np.ndarray:
    """Rotation that turns the vector `up` onto world z."""
    axis = np.cross(up, [0, 0, 1.0])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    return rot_about(axis / s * np.arctan2(s, up[2]))

def cop_xy(p_ee: np.ndarray, dw: np.ndarray, z_contact: float) -> np.ndarray | None:
    """World xy of the centre of pressure on the plane z = z_contact, from a world wrench change about the EE point
    (horizontal friction forces corrected through the lever arm)."""
    if dw[2] < 1.0:
        return None
    rz = z_contact - p_ee[2]
    return p_ee[:2] + np.array([rz * dw[0] - dw[4], dw[3] + rz * dw[1]]) / dw[2]

class WipeExpert(Base):
    """Hold the wiping pad level, lower it over the start of the marker path until the wrist feels the surface, then
    follow the path while setting the height every step from the normal force (toward 14 N: shallower presses leave
    the soft pad uneven) and turning the pad toward its centre of pressure (flush with the surface). Native: lower the
    pad slowly onto the drawn top (a fast descent hit it with over 40 N), then wipe at a fixed depth below it (at
    6 mm the press fell to about 9.5 N with the arm stretched, too little to press the soft pad evenly).
    Steps along the path and in depth are limited separately, so pressing never takes the step of the path motion."""

    APPROACH, DESCEND, WIPE, LIFT = range(4)
    F_TARGET, F_CONTACT, GAIN, SPEED, TILT_GAIN, MAX_TILT = 14.0, 2.0, 0.0003, 0.01, 0.1, 0.1
    TILT_STEP = 0.002  # rad per step: the pad is stiff to tilt (0.5 deg puts the centre of pressure at its end)
    XY_STEP, Z_STEP, Z_LEAD = 0.012, 0.015, 0.015  # m per step along the path and in depth; set-point lead in depth
    LEAD = 0.02  # m, the path point runs at most this far ahead of the pad
    NATIVE_DEPTH = 0.008  # native: wipe this far below the drawn top

    def __init__(self, env, rng):
        super().__init__(env, rng)
        sim = env.sim
        markers = env.model.mujoco_arena.markers
        self.path = np.array([sim.data.body_xpos[sim.model.body_name2id(m.root_body)][:2] for m in markers])
        self.table_z = float(sim.data.body_xpos[sim.model.body_name2id(markers[0].root_body)][2])  # drawn top
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        s = np.arange(-0.02, d[-1] + 0.02, 0.004)
        u0 = (self.path[1] - self.path[0]) / max(d[1], 1e-9)
        u1 = (self.path[-1] - self.path[-2]) / max(d[-1] - d[-2], 1e-9)
        self.dense = np.array([
            self.path[0] + u0 * si if si < 0 else self.path[-1] + u1 * (si - d[-1]) if si > d[-1]
            else np.array([np.interp(si, d, self.path[:, 0]), np.interp(si, d, self.path[:, 1])]) for si in s])
        _, rs = env.tool_frame()  # pad level, long side across the path's overall direction
        level = tilt_to_level(rs[:, 2])
        long = level @ rs[:, 0]
        heading = self.dense[-1] - self.dense[0]
        dyaw = np.arctan2(heading[0], -heading[1]) - np.arctan2(long[1], long[0])
        dyaw = (dyaw + np.pi / 2) % np.pi - np.pi / 2
        self.rot0 = sq.rot_z(dyaw) @ level @ np.array(self.ctrl.ee_ori_mat)
        self.rot, self.tilt = self.rot0, np.zeros(3)  # tilt: rotation vector added to the level pose, bounded
        self.progress, self.z_tgt, self.sweeps = 0.0, None, 0

    def act(self, block):
        p, r = self.ee()
        bottom, _ = self.env.tool_frame()
        noise = 0.008
        self.t_phase += 1
        if self.phase == self.APPROACH:
            tgt = p + np.r_[self.dense[0] - bottom[:2], self.table_z + 0.05 - bottom[2]]
            a = track(p, r, tgt, self.rot, 0.03)
            if np.linalg.norm(tgt - p) < 0.008 and np.linalg.norm(axisangle(self.rot @ r.T)) < 0.04:
                self.tare = world_wrench(r, block)
                self.go(self.DESCEND)
            self.failed = self.t_phase > 100
        elif self.phase == self.DESCEND:
            f = world_wrench(r, block)[2] - self.tare[2]
            if self.native and bottom[2] < self.table_z + 0.002:
                self.z_tgt = p[2] + (self.table_z - self.NATIVE_DEPTH) - bottom[2]
                self.go(self.WIPE)
            elif f > self.F_CONTACT:
                self.z_tgt = p[2] - 0.002
                self.go(self.WIPE)
            step = 0.01 if bottom[2] > self.table_z + 0.03 else 0.003  # slow near the surface
            a = track(p, r, p + np.r_[self.dense[0] - bottom[:2], -step], self.rot, 0.01)
            self.failed = self.t_phase > 80
        elif self.phase == self.WIPE:
            noise = 0.004
            if not self.native:
                dw = world_wrench(r, block) - self.tare
                self.z_tgt -= float(np.clip(self.GAIN * (self.F_TARGET - dw[2]), -0.002, 0.002))
                self.z_tgt = float(np.clip(self.z_tgt, p[2] - self.Z_LEAD, p[2] + self.Z_LEAD))
                c = cop_xy(p, dw, bottom[2])
                if c is not None and dw[2] > 4.0:  # lift the side that presses harder
                    off = c - bottom[:2]
                    self.tilt = np.clip(self.tilt + np.clip(self.TILT_GAIN * np.array([off[1], -off[0], 0.0]), -self.TILT_STEP, self.TILT_STEP),
                                        -self.MAX_TILT, self.MAX_TILT)
                    self.rot = rot_about(self.tilt) @ self.rot0
            if np.linalg.norm(self.dense[int(self.progress)] - bottom[:2]) < self.LEAD:  # the path waits for the pad
                self.progress = min(self.progress + self.SPEED / 0.004, len(self.dense) - 1)
            xy = self.dense[int(self.progress)]
            dxy = xy - bottom[:2]
            dxy *= min(1.0, self.XY_STEP / max(np.linalg.norm(dxy), 1e-9))
            rv = axisangle(self.rot @ r.T)
            rv *= min(1.0, 0.15 / max(np.linalg.norm(rv), 1e-9))
            dz = np.clip(self.z_tgt - p[2], -self.Z_STEP, self.Z_STEP)
            a = np.concatenate([np.clip(np.r_[dxy, dz] / 0.05, -1, 1), np.clip(rv / 0.5, -1, 1)])
            if self.progress >= len(self.dense) - 1 and np.linalg.norm(bottom[:2] - xy) < 0.01:
                if len(self.env.wiped_markers) == self.env.num_markers:
                    self.go(self.LIFT)
                else:  # missed some: sweep back once
                    self.failed = self.sweeps > 0
                    self.dense, self.progress, self.sweeps = self.dense[::-1].copy(), 0.0, self.sweeps + 1
        else:
            a = track(p, r, p + [0, 0, 0.03], self.rot, 0.02)
        return self.finish(a, -1.0, noise)  # 7-D actions; the pad ignores the last entry

class DoorExpert:
    """Opens robosuite's latched door. It grasps the middle of the handle bar from the front (gripper pointing at the
    door, fingers vertical), turns the handle down about the latch axis until the latch clears the frame, then pulls:
    the hand pose of the turn's end, turned on to TURN_LEAD so that the OSC keeps pressing the handle against its
    spring, is carried about the hinge (the right post) by a door angle that runs at most 0.2 rad ahead of the hand."""

    APPROACH, DESCEND, GRASP, ARC, TURN = 0, 1, 2, 4, 5  # the phase labels of the datasets
    HINGE_OFFSET = 0.255  # m, hinge axis from the panel center
    ARC_STEP = 0.04  # rad per step along the hinge arc
    TURN_RAD, TURN_STEP, TURN_LEAD = 1.3, 0.06, 1.8  # latch angle that frees the door (measured); rad per step; target

    def __init__(self, env, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.ctrl = env.robots[0].controller
        sim = env.sim
        self.handle_site = sim.model.site_name2id("Door_handle")
        r_door = sim.data.body_xmat[sim.model.body_name2id(env.door.door_body)].reshape(3, 3)
        pull = -r_door[:, 1].copy()
        pull[2] = 0
        self.pull = pull / np.linalg.norm(pull)  # away from the panel, toward the robot
        self.finger_axis = finger_axis(env)
        self.bar = r_door[:, 0]  # along the handle bar
        rng.choice([-1.0, 1.0]), rng.uniform(0.045, 0.055)  # draws kept so that the dataset seeds give their demos
        self.grip_offset = -0.05  # the middle of the stock bar, which runs from its base to the handle site
        self.phase, self.t_phase, self.failed = self.APPROACH, 0, False

    def _rot(self, pull):
        """Gripper pointing at the door (against `pull`) with vertical fingers. The finger sense is fixed: the other
        one bent wrist joint 6 into its limit when the door faced the robot."""
        f, z = np.array([0, 0, -1.0]), -pull / np.linalg.norm(pull)
        cols = [None, None, z]
        cols[self.finger_axis] = f
        cols[1 - self.finger_axis] = np.cross(f, z) if self.finger_axis == 1 else np.cross(z, f)
        return np.stack(cols, axis=1)

    def _go(self, phase):
        self.phase, self.t_phase = phase, 0

    def _set_hinge(self):
        """Hinge axis of the right post (door geometry) and the rotation sense that brings the hand toward the robot."""
        door = self.env.sim.model.body_name2id(self.env.door.door_body)
        r_door = self.env.sim.data.body_xmat[door].reshape(3, 3)
        self.hinge = np.array(self.env.sim.data.body_xpos[door]) + r_door @ np.array([self.HINGE_OFFSET, 0, 0])
        rad = np.array(self.ctrl.ee_pos) - self.hinge
        self.turn = 1.0 if np.cross([0, 0, 1.0], rad) @ self.pull > 0 else -1.0

    def _latch(self):
        """Latch origin, its axis in the sense that turns the handle down, and the handle angle in that sense."""
        env = self.env
        i = env.sim.model.body_name2id(env.door.latch_body)
        o, y = np.array(env.sim.data.body_xpos[i]), np.array(env.sim.data.body_xmat[i]).reshape(3, 3)[:, 1]
        return o, self.sense * y, self.sense * float(env.sim.data.qpos[env.handle_qpos_addr])

    def start_pull(self):
        p, r = np.array(self.ctrl.ee_pos), np.array(self.ctrl.ee_ori_mat)
        o, axis, th = self._latch()
        turn = rot_about(axis * (self.TURN_LEAD - th))
        self.hold, self.phi, self.comp = (o + turn @ (p - o), turn @ r), 0.0, np.zeros(3)
        self._go(self.ARC)

    def act(self, block):
        env, p, r = self.env, np.array(self.ctrl.ee_pos), np.array(self.ctrl.ee_ori_mat)
        if self.phase == self.TURN:
            self.t_phase += 1
            o, axis, th = self._latch()
            self.angle = min(self.angle + self.TURN_STEP, self.TURN_LEAD)
            turn = rot_about(axis * self.angle)
            p0, r0 = self.grasp_pose
            a = track(p, r, o + turn @ (p0 - o), turn @ r0, 0.02)
            if th >= self.TURN_RAD:
                self.start_pull()
            self.failed = self.t_phase > 100
        elif self.phase == self.ARC:
            self.t_phase += 1
            hp, hr = self.hold
            v0, v1 = (hp - self.hinge)[:2], (p - self.hinge)[:2]
            phi_hand = self.turn * np.arctan2(v0[0] * v1[1] - v0[1] * v1[0], v0 @ v1)
            self.phi = min(self.phi + self.ARC_STEP, phi_hand + 0.2)
            turn = rot_about(np.array([0.0, 0.0, self.turn * self.phi]))
            a = track(p, r, self.hinge + turn @ (hp - self.hinge) + self.comp, turn @ hr, 0.03)
            self.failed = self.t_phase > 150
        else:
            handle = np.array(env.sim.data.site_xpos[self.handle_site])
            grip = -1.0
            self.t_phase += 1
            grasp = handle + self.grip_offset * self.bar
            if self.phase == self.APPROACH:  # in front of the bar
                r_tgt = self._rot(self.pull)
                tgt = grasp + 0.10 * self.pull
                a = track(p, r, tgt, r_tgt, 0.03)
                aligned = np.linalg.norm(axisangle(r_tgt @ r.T)) < (0.04 if self.t_phase < 60 else 0.1)
                if np.linalg.norm(tgt - p) < 0.008 and aligned:
                    self._go(self.DESCEND)
                self.failed = self.t_phase > 120
            elif self.phase == self.DESCEND:  # forward onto the bar
                a = track(p, r, grasp, self._rot(self.pull), 0.01)
                if np.linalg.norm(grasp - p) < 0.004:
                    self._go(self.GRASP)
                self.failed = self.t_phase > 60
            else:  # GRASP
                grip = 1.0
                a = track(p, r, p, r, 0.01)
                if self.t_phase >= 10:  # grasped: turn the handle first
                    self._set_hinge()
                    i = env.sim.model.body_name2id(env.door.latch_body)
                    o, y = np.array(env.sim.data.body_xpos[i]), np.array(env.sim.data.body_xmat[i]).reshape(3, 3)[:, 1]
                    self.sense = -1.0 if np.cross(y, np.array(env.sim.data.site_xpos[self.handle_site]) - o)[2] > 0 else 1.0
                    self.grasp_pose, self.angle = (p, r), 0.0
                    self._go(self.TURN)
            a = a + 0.01 * self.rng.standard_normal(6)
            return np.concatenate([np.clip(a, -1, 1), [grip]])
        a = a + 0.005 * self.rng.standard_normal(6)
        return np.concatenate([np.clip(a, -1, 1), [1.0]])

EXPERTS = {"stack_ft": StackExpert, "wipe_ft": WipeExpert, "door_ft": DoorExpert}
