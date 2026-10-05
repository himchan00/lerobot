"""Reference impedance-ODE model of robosuite 1.4 OSC_POSE (the LIBERO controller), in numpy.

Use it as the ground-truth simulator for task 2 (recovering x_d, kp, kd) and as the spec for a
differentiable torch version. Inputs are one frame of the regenerated dataset (`regenerate_libero_hf.py`).

Model C (validated on robosuite Lift, see HANDOFF): per physics substep k = 0..24, dt = 2 ms,
    f*  = [kp_p (x_d - x) - kd_p v ;  kp_o e_ori(R_d, R) - kd_o w]          (OSC command, accel units)
    a   = Lambda_full^-1 (Lambda_blk f* + W_k)                              (6-D task-space accel)
    v  += dt a_lin ; w += dt a_ang ; x += dt v ; R = Exp(dt w) R             (semi-implicit Euler, world frame)
with Lambda from the frame start, W_k the external wrench about the EE point at the start of substep k
(`hf.contact_wrench_ee` = ground truth, `hf.ft_wrench_ee` = gravity-compensated F/T), and e_ori the
robosuite `orientation_error` (0.5 * sum_i r_i x r_d,i, not the log map). With uncoupled OSC,
Lambda_blk = blockdiag(Lambda_pos, Lambda_ori) != Lambda_full, which couples rotation into translation.
"""

from __future__ import annotations

import numpy as np

DT = 0.002


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """robosuite.utils.control_utils.orientation_error."""
    return 0.5 * sum(np.cross(current[:, i], desired[:, i]) for i in range(3))


def exp_so3(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def simulate_frame(
    frame: dict,
    x_d: np.ndarray | None = None,
    R_d: np.ndarray | None = None,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    wrench_key: str | None = "hf.contact_wrench_ee",
    model: str = "C",
    n_sub: int = 25,
) -> dict:
    """Roll the OSC closed loop over one control step from the logged frame-start state.

    Parameters default to the logged ground truth (`osc.*`); pass estimates to score them.
    model: "A" unit mass, no wrench | "B" block Lambda + wrench | "C" full Lambda + wrench.
    Returns world-frame positions, rotations and velocities at time points 1..n_sub.
    """
    x_d = frame["osc.goal_pos"] if x_d is None else x_d
    R_d = quat_xyzw_to_mat(frame["osc.goal_quat"]) if R_d is None else R_d
    kp = frame["osc.kp"] if kp is None else np.broadcast_to(kp, (6,))
    kd = frame["osc.kd"] if kd is None else np.broadcast_to(kd, (6,))
    lam_full = frame["osc.lambda_full"]
    lam_blk = np.zeros((6, 6))
    lam_blk[:3, :3], lam_blk[3:, 3:] = frame["osc.lambda_pos"], frame["osc.lambda_ori"]
    lf_inv, lb_inv = np.linalg.inv(lam_full), np.linalg.inv(lam_blk)

    x = np.array(frame["hf.ee_pos"][0], dtype=np.float64)
    R = quat_xyzw_to_mat(frame["hf.ee_quat"][0])
    v = np.array(frame["hf.ee_linvel"][0], dtype=np.float64)
    w = np.array(frame["hf.ee_angvel"][0], dtype=np.float64)
    xs, Rs, vs, ws = [], [], [], []
    for k in range(n_sub):
        fs = np.concatenate([kp[:3] * (x_d - x) - kd[:3] * v, kp[3:] * orientation_error(R_d, R) - kd[3:] * w])
        wr = np.zeros(6) if (wrench_key is None or model == "A") else frame[wrench_key][k]
        if model == "A":
            acc = fs
        elif model == "B":
            acc = fs + lb_inv @ wr
        else:
            acc = lf_inv @ (lam_blk @ fs + wr)
        v = v + DT * acc[:3]
        w = w + DT * acc[3:]
        x = x + DT * v
        R = exp_so3(DT * w) @ R
        xs.append(x.copy()), Rs.append(R.copy()), vs.append(v.copy()), ws.append(w.copy())
    return {"pos": np.stack(xs), "rot": np.stack(Rs), "linvel": np.stack(vs), "angvel": np.stack(ws)}


def position_error(frame: dict, sim: dict) -> np.ndarray:
    """Per-time-point position error (m) against the logged trajectory, time points 1..N."""
    return np.linalg.norm(sim["pos"] - frame["hf.ee_pos"][1:], axis=-1)


def to_ee_frame(frame: dict, key: str = "hf.ft_wrench_ee") -> np.ndarray:
    """World-frame wrenches about the EE point -> EE (body) frame: [R^T F, R^T tau], per time point."""
    out = []
    for q, wr in zip(frame["hf.ee_quat"], frame[key]):
        Rt = quat_xyzw_to_mat(q).T
        out.append(np.concatenate([Rt @ wr[:3], Rt @ wr[3:]]))
    return np.stack(out)
