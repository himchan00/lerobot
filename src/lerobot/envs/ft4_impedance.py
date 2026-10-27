"""Tier-3 impedance control for FT-4: robosuite's OSC_POSE with a per-tick reference and a feedforward wrench.

Each Tier-2 tick (one 20 Hz env step, h = 0.05 s) gives the next two EE poses x̂_{t+1}, x̂_{t+2} and the tick-mean
wrenches W̄_{t+1}, W̄_{t+2} of the environment on the hand (about the EE point) over the next two ticks, all in the
frame of the measured EE pose x_t (the grip site's, like the controller and the F/T observation). Over the tick, the
reference x_d(τ), τ ∈ [0, h], is a quintic from the measured state (pose, velocity, and the mean acceleration over the
last tick) to x̂_{t+1}; its end velocity and acceleration are central differences of x_t, x̂_{t+1} and x̂_{t+2}.
Rotation takes the quintic in the chart φ = log(R_tᵀR) of the measured R_t, with ω, ω̇ entering it as φ̇(0) = R_tᵀω,
φ̈(0) = R_tᵀω̇ (exact at φ = 0). A tick mean is the value at mid-tick, so F_d(τ) is piecewise linear through W̄_{t+1}
at τ = h/2, from the measured W̄_t (`wbar`, set by the env every tick) to (W̄_{t+1} + W̄_{t+2})/2. Every physics step
commands

    F = Λ(ẍ_d − k_d ė − k_p e) − F_d,    e = x − x_d  (rotation: R_d·log(R_dᵀR), ė = ω − ω_d),

with OSC's gains and the full Λ (OSC's `uncouple_pos_ori` blocks would leak the feedforward acceleration between
position and rotation: a 0.5 m/s² free-space residual, a 3-7 mm lag at kp 150), plus robosuite's gravity/Coriolis and
null-space torques and torques that cancel the arm model's joint damping and friction loss, so that
Λ(q) ë + Λk_d ė + Λk_p e = F_ext − F_d (M = Λ(q), no wrench feedback). The first tick after a reset extrapolates the
tick before it at constant velocity (mirroring x̂_{t+1}) for the start velocity and acceleration; F_d starts at
W̄_{t+1}. FT-4's first tick is a settle step (a hold): at rest, F_d 0.
"""

from __future__ import annotations

import numpy as np
import robosuite.robots.single_arm as _single_arm
from robosuite.controllers.base_controller import Controller
from robosuite.controllers.osc import OperationalSpaceController
from robosuite.utils.control_utils import nullspace_torques, opspace_matrices
from scipy.spatial.transform import Rotation

IMPEDANCE = "OSC_IMPEDANCE"  # controller type in a robosuite controller config
FRICTION_VEL = 0.01  # rad/s: the friction-loss compensation saturates over this joint speed


def _exp(v: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(v).as_matrix()


def _log(r: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(r).as_rotvec()


def quintic(x0, v0, a0, x1, v1, a1, h: float) -> np.ndarray:
    """(6, d) coefficients, lowest order first, of the quintic on [0, h] with these end values and derivatives."""
    d = x1 - x0
    c3 = (20 * d - (8 * v1 + 12 * v0) * h - (3 * a0 - a1) * h**2) / (2 * h**3)
    c4 = (-30 * d + (14 * v1 + 16 * v0) * h + (3 * a0 - 2 * a1) * h**2) / (2 * h**4)
    c5 = (12 * d - 6 * (v1 + v0) * h + (a1 - a0) * h**2) / (2 * h**5)
    return np.stack([x0, v0, a0 / 2, c3, c4, c5])


def evaluate(c: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value, first and second derivative of the quintic c at tau."""
    p = tau ** np.arange(6)
    return p @ c, (np.arange(1, 6) * p[:5]) @ c[1:], (np.array([2, 6, 12, 20]) * p[:4]) @ c[2:]


def angular_velocity(phi, dphi, ddphi) -> tuple[np.ndarray, np.ndarray]:
    """ω = J_l(φ)φ̇ and ω̇ of Exp(φ(τ)), in the chart's base frame. J_l(φ) = I + a[φ]× + b[φ]×²."""
    t2 = phi @ phi
    if t2 < 1e-4:  # series below 0.01 rad; da, db are a'(θ)/θ, b'(θ)/θ
        a, b, da, db = 0.5 - t2 / 24, 1 / 6 - t2 / 120, -1 / 12 + t2 / 180, -1 / 60 + t2 / 1260
    else:
        t = np.sqrt(t2)
        s, c = np.sin(t), np.cos(t)
        a, b = (1 - c) / t2, (t - s) / (t2 * t)
        da, db = (t * s - 2 * (1 - c)) / t2**2, (t * (1 - c) - 3 * (t - s)) / (t2**2 * t)
    cross = np.cross(phi, dphi)
    w = dphi + a * cross + b * np.cross(phi, cross)
    dot = phi @ dphi
    dw = (ddphi + a * np.cross(phi, ddphi) + b * np.cross(phi, np.cross(phi, ddphi))
          + da * dot * cross + db * dot * np.cross(phi, cross) + b * np.cross(dphi, cross))
    return w, dw


class ImpedanceOSC(OperationalSpaceController):
    """OSC_POSE driven by the Tier-3 reference. Arm action (24): [p̂_{t+1}, r̂_{t+1}, p̂_{t+2}, r̂_{t+2}, W̄_{t+1}, W̄_{t+2}]
    in the current EE frame: p̂ = R_tᵀ(p − p_t), r̂ = log(R_tᵀR), the wrenches [f, m] rotated by R_tᵀ; m, rad, N, N·m."""

    def __init__(self, interpolation=None, **kwargs):
        if interpolation is not None:
            raise ValueError("The impedance controller makes its own reference; set interpolation to None.")
        self.seg = None  # the current tick's reference; None until the first tick after a reset
        self.wbar = None  # measured W̄_t (world), set by the env before each tick and used once
        super().__init__(**kwargs)
        self.control_dim = 24
        self.input_min, self.input_max = np.full(24, -np.inf), np.full(24, np.inf)
        self.h = 1.0 / self.control_freq
        self.dt = self.sim.model.opt.timestep

    def reset_goal(self):
        super().reset_goal()
        self.seg = None

    def reference(self, seg: dict, tau: float):
        """x_d, ẋ_d, ẍ_d at tau: position, velocity, acceleration, R_d, world ω_d, world ω̇_d."""
        p, v, a = evaluate(seg["pos"], tau)
        phi, dphi, ddphi = evaluate(seg["rot"], tau)
        w, dw = angular_velocity(phi, dphi, ddphi)
        r0 = seg["r0"]
        return p, v, a, r0 @ _exp(phi), r0 @ w, r0 @ dw

    def wrench(self, seg: dict, tau: float) -> np.ndarray:
        """F_d at tau: linear from the start value to W̄_{t+1} at h/2, then to the end value."""
        s = 2 * tau / self.h
        return seg["f0"] + (seg["fm"] - seg["f0"]) * s if s <= 1 else seg["fm"] + (seg["f1"] - seg["fm"]) * (s - 1)

    def command(self, seg: dict, tau: float):
        """Commanded wrench F at the current state for reference `seg` at tau, and the opspace matrices."""
        pd, vd, ad, rd, wd, dwd = self.reference(seg, tau)
        e = np.r_[self.ee_pos - pd, rd @ _log(rd.T @ self.ee_ori_mat)]
        de = np.r_[self.ee_pos_vel - vd, self.ee_ori_vel - wd]
        lam = opspace_matrices(self.mass_matrix, self.J_full, self.J_pos, self.J_ori)
        return lam[0] @ (np.r_[ad, dwd] - self.kd * de - self.kp * e) - self.wrench(seg, tau), lam

    def set_goal(self, action, set_pos=None, set_ori=None):
        self.update()
        p, r, h = np.array(self.ee_pos), np.array(self.ee_ori_mat), self.h
        v, w = np.array(self.ee_pos_vel), np.array(self.ee_ori_vel)
        wbar, self.wbar = self.wbar, None
        p1, p2 = p + r @ action[0:3], p + r @ action[6:9]
        r1, r2 = r @ _exp(action[3:6]), r @ _exp(action[9:12])
        w1, w2 = (np.r_[r @ action[i : i + 3], r @ action[i + 3 : i + 6]] for i in (12, 18))
        if self.seg is None:
            # No previous tick: the tick before is the constant-velocity mirror of x̂_{t+1} about the measured pose
            # (as the labels extrapolate an episode's last frame, geometry._extend_past_end), whose central
            # differences give the start velocity and acceleration; F_d starts at W̄_{t+1}.
            p_back, r_back = p + r @ r1.T @ (p - p1), r @ r1.T @ r
            v0, a0 = (p1 - p_back) / (2 * h), (p_back - 2 * p + p1) / h**2
            q_back, q_next = _log(r.T @ r_back), _log(r.T @ r1)
            w0, dw0 = (q_next - q_back) / (2 * h), (q_back + q_next) / h**2
            f0 = w1
        else:  # the measured velocity and the mean acceleration over the last tick; F_d at the measured W̄_t
            v0, a0 = v, (v - self.seg["v"][:3]) / h
            w0, dw0 = r.T @ w, r.T @ (w - self.seg["v"][3:]) / h
            f0 = self.seg["f1"] if wbar is None else wbar
        q1, q2 = _log(r.T @ r1), _log(r.T @ r2)
        self.seg = {
            "r0": r, "f0": f0, "fm": w1, "f1": (w1 + w2) / 2, "i": 0, "v": np.r_[v, w],
            "pos": quintic(p, v0, a0, p1, (p2 - p) / (2 * h), (p - 2 * p1 + p2) / h**2, h),
            "rot": quintic(np.zeros(3), w0, dw0, q1, q2 / (2 * h), (q2 - 2 * q1) / h**2, h),
        }

    def joint_losses(self) -> np.ndarray:
        """Torques cancelling the arm's joint damping (MuJoCo's passive force) and friction loss (smoothed through zero
        speed), which the gravity/Coriolis compensation leaves out."""
        i = self.qvel_index
        friction = self.sim.model.dof_frictionloss[i] * np.tanh(self.joint_vel / FRICTION_VEL)
        return -np.array(self.sim.data.qfrc_passive[i]) + friction

    def run_controller(self):
        self.update()
        tau = self.seg["i"] * self.dt
        self.seg["i"] += 1
        wrench, lam = self.command(self.seg, tau)
        self.torques = self.J_full.T @ wrench + self.torque_compensation + self.joint_losses()
        self.torques += nullspace_torques(self.mass_matrix, lam[3], self.initial_joint, self.joint_pos, self.joint_vel)
        Controller.run_controller(self)
        return self.torques


_stock_factory = _single_arm.controller_factory


def _controller_factory(name, params):
    return ImpedanceOSC(**params) if name == IMPEDANCE else _stock_factory(name, params)


_single_arm.controller_factory = _controller_factory  # robosuite builds a robot's controller by type name
