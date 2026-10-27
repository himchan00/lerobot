# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Body product geometry for LIBERO pose control.

pose7 is ``[position_xyz, quaternion_xyzw]``. Raw observations remain LIBERO
state8 ``[position_xyz, rotvec_xyz, finger_qpos_2]``.

The body-space six-vectors handled by this module are normalized controller
increments: translation entries are command units where 1 -> 0.05 m, and
rotation entries are command units where 1 -> 0.5 rad. ``local()`` converts a
pose delta into those normalized body increments, and ``retract()`` applies
them back to a pose. Preserve float64, otherwise use float32.
"""

import torch
from torch import Tensor

__all__ = [
    "IMPEDANCE_GROUPS",
    "action_to_endpoint",
    "body_to_world",
    "hand_to_grip_site",
    "impedance_action",
    "impedance_next_state",
    "impedance_one_step",
    "local",
    "nearest_pose_indices",
    "perturb_pose_state",
    "pose_from_state",
    "pose_state_features",
    "retract",
    "tick_mean_wrench",
]


_LIBERO_TRANSLATION_PER_COMMAND = 0.05
_LIBERO_ANGULAR_PER_COMMAND = 0.5
# robosuite mounts its Panda grippers turned −90° about the hand's z: R_grip_site = R_hand @ Rz(−90°) (xyzw)
_GRIP_SITE_IN_HAND = (0.0, 0.0, -(0.5**0.5), 0.5**0.5)


def _promote(*values: Tensor) -> tuple[Tensor, ...]:
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in values) else torch.float32
    return tuple(value.to(dtype=dtype) for value in values)


def _broadcast(*values: Tensor) -> tuple[Tensor, ...]:
    shape = torch.broadcast_shapes(*(value.shape[:-1] for value in values))
    return tuple(value.expand(*shape, value.shape[-1]) for value in values)


def _unit(quaternion: Tensor) -> Tensor:
    scaled = quaternion / quaternion.abs().amax(dim=-1, keepdim=True)
    return scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)


def _conjugate(quaternion: Tensor) -> Tensor:
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)


def _exp(rotvec: Tensor) -> Tensor:
    # Scaling also keeps the norm finite for very large finite rotation vectors.
    scale = rotvec.abs().amax(dim=-1, keepdim=True).clamp_min(1)
    scaled = rotvec / scale
    norm_squared = scaled.square().sum(dim=-1, keepdim=True)
    small = norm_squared < 1e-6
    norm = torch.sqrt(torch.where(small, torch.ones_like(norm_squared), norm_squared))
    half_angle = (0.5 * scale) * norm
    squared = torch.where(small, norm_squared, torch.zeros_like(norm_squared))
    factor = 0.5 - squared / 48 + squared.square() / 3840 - squared.pow(3) / 645120
    scalar = 1 - squared / 8 + squared.square() / 384 - squared.pow(3) / 46080
    vector = torch.where(small, rotvec * factor, scaled * (half_angle.sin() / norm))
    scalar = torch.where(small, scalar, half_angle.cos())
    return torch.cat((vector, scalar), dim=-1)


def _log(quaternion: Tensor) -> Tensor:
    vector, scalar = quaternion[..., :3], quaternion[..., 3:]
    pivot = vector.gather(-1, vector.abs().argmax(dim=-1, keepdim=True))
    flip = (scalar < 0) | ((scalar == 0) & (pivot < 0))
    vector = torch.where(flip, -vector, vector)
    scalar = torch.where(flip, -scalar, scalar)
    norm_squared = vector.square().sum(dim=-1, keepdim=True)
    small = norm_squared < 1e-6
    norm = torch.sqrt(torch.where(small, torch.ones_like(norm_squared), norm_squared))
    squared = torch.where(small, norm_squared, torch.zeros_like(norm_squared))
    factor = 2 + squared / 3 + 3 * squared.square() / 20 + 5 * squared.pow(3) / 56
    factor = torch.where(small, factor, 2 * torch.atan2(norm, scalar) / norm)
    return vector * factor


def _multiply(left: Tensor, right: Tensor) -> Tensor:
    left_vector, left_scalar = left[..., :3], left[..., 3:]
    right_vector, right_scalar = right[..., :3], right[..., 3:]
    vector = (
        left_scalar * right_vector
        + right_scalar * left_vector
        + torch.linalg.cross(left_vector, right_vector, dim=-1)
    )
    scalar = left_scalar * right_scalar - (left_vector * right_vector).sum(dim=-1, keepdim=True)
    return _unit(torch.cat((vector, scalar), dim=-1))


def _matrix(quaternion: Tensor) -> Tensor:
    x, y, z, w = quaternion.unbind(dim=-1)
    xx, yy, zz = x.square(), y.square(), z.square()
    xy, xz, yz, xw, yw, zw = x * y, x * z, y * z, x * w, y * w, z * w
    return torch.stack(
        (
            1 - 2 * (yy + zz),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (xx + zz),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (xx + yy),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def pose_from_state(state: Tensor) -> Tensor:
    """Convert raw state8 to position and unit xyzw quaternion, excluding fingers."""
    (state,) = _promote(state)
    with torch.autocast(device_type=state.device.type, enabled=False):
        return torch.cat((state[..., :3], _unit(_exp(state[..., 3:6]))), dim=-1)


def hand_to_grip_site(state: Tensor) -> Tensor:
    """Turn state8's rotation from robosuite's eef_quat (hand body) to the grip site's, the frame of the controller
    and the F/T. The position is the grip site's already."""
    (state,) = _promote(state)
    with torch.autocast(device_type=state.device.type, enabled=False):
        mount = state.new_tensor(_GRIP_SITE_IN_HAND).expand(*state.shape[:-1], 4)
        rotation = _log(_multiply(_exp(state[..., 3:6]), mount))
        return torch.cat((state[..., :3], rotation, state[..., 6:]), dim=-1)


def local(pose: Tensor, target_pose: Tensor) -> Tensor:
    """Return normalized body controller increments.

    The output is ``[R.T @ (p_target - p) / 0.05, Log(R.T @ R_target) / 0.5]``.

    The rotational inverse is principal, not uniquely smooth at the pi cut.
    """
    pose, target_pose = _broadcast(*_promote(pose, target_pose))
    with torch.autocast(device_type=pose.device.type, enabled=False):
        quaternion = _unit(pose[..., 3:])
        translation = (
            _matrix(quaternion).transpose(-1, -2) @ (target_pose[..., :3] - pose[..., :3])[..., None]
        )
        rotation = _log(_multiply(_conjugate(quaternion), _unit(target_pose[..., 3:])))
        return torch.cat(
            (
                translation.squeeze(-1) / _LIBERO_TRANSLATION_PER_COMMAND,
                rotation / _LIBERO_ANGULAR_PER_COMMAND,
            ),
            dim=-1,
        )


def retract(pose: Tensor, increment: Tensor) -> Tensor:
    """Apply normalized body controller increments to a pose.

    ``increment[..., :3]`` is interpreted in 0.05 m command units and
    ``increment[..., 3:]`` in 0.5 rad command units. Rotation composes on the
    right, and ``local``/``retract`` are inverse for ``||0.5 * u_rot|| < pi``.
    """
    pose, increment = _broadcast(*_promote(pose, increment))
    with torch.autocast(device_type=pose.device.type, enabled=False):
        quaternion = _unit(pose[..., 3:])
        position = pose[..., :3] + (
            _matrix(quaternion) @ (_LIBERO_TRANSLATION_PER_COMMAND * increment[..., :3, None])
        ).squeeze(-1)
        return torch.cat(
            (
                position,
                _multiply(quaternion, _exp(_LIBERO_ANGULAR_PER_COMMAND * increment[..., 3:])),
            ),
            dim=-1,
        )


def _rotate_triplets(rotation: Tensor, vectors: Tensor) -> Tensor:
    """(..., 6) as two 3-vectors (e.g. a wrench [f, m]), each rotated by (..., 3, 3)."""
    return (rotation @ vectors.unflatten(-1, (2, 3)).transpose(-1, -2)).transpose(-1, -2).flatten(-2)


def body_to_world(pose: Tensor, vectors: Tensor) -> Tensor:
    """Rotate both normalized controller triplets by R."""
    pose, vectors = _broadcast(*_promote(pose, vectors))
    with torch.autocast(device_type=pose.device.type, enabled=False):
        return _rotate_triplets(_matrix(_unit(pose[..., 3:])), vectors)


# Tier-3 outputs [x̂_{t+1} pos, rot | W̄_{t+1} force, torque | contact logit | gripper logit] (W̄: the tick-mean
# wrench): one σ² / scale group per Gaussian 3-vector
IMPEDANCE_GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def _extend_past_end(state: Tensor, real: Tensor) -> tuple[Tensor, Tensor]:
    """A (..., T) state8 window with the frame just past an episode's end (``real`` marks frames inside it)
    extrapolated at constant body velocity from the last two frames (fingers held), and the mask of the frames now
    known. A window that starts at the last frame cannot extrapolate: its frame 1 stays the copy and unknown."""
    (state,) = _promote(state)
    with torch.autocast(device_type=state.device.type, enabled=False):
        pose = pose_from_state(state)
        prev, prev2 = pose[..., 1:-1, :], pose[..., :-2, :]
        ahead = retract(prev, local(prev2, prev))  # frame j (2..T-1) at constant velocity from j-1, j-2
        ahead = torch.cat((ahead[..., :3], _log(ahead[..., 3:]), state[..., 1:-1, 6:]), dim=-1)
        past_end = real[..., 1:-1] & ~real[..., 2:]  # frame j is the first past the end
        state = torch.cat((state[..., :2, :], torch.where(past_end[..., None], ahead, state[..., 2:, :])), dim=-2)
        return state, real | torch.cat((torch.zeros_like(real[..., :2]), past_end), dim=-1)


def tick_mean_wrench(state: Tensor, ft: Tensor, previous: Tensor | None = None) -> Tensor:
    """Each frame's tick-mean wrench (the mean of its (..., T, N, 6) F/T samples) in its own grip-site frame. Sample j
    of N is logged in the frame of its physics step, taken as the body slerp from the previous frame's at j/N; frame 0
    takes it from ``previous`` (..., 1, 8) when given, else the plain mean. state (..., T, 8) → (..., T, 6)."""
    frames = state if previous is None else torch.cat((previous, state), dim=-2)
    quaternion = _unit(pose_from_state(frames)[..., 3:])
    phi = _log(_multiply(_conjugate(quaternion[..., :-1, :]), quaternion[..., 1:, :]))  # G_{i-1}ᵀ G_i
    if previous is None:
        phi = torch.cat((torch.zeros_like(phi[..., :1, :]), phi), dim=-2)
    n = ft.shape[-2]
    frac = torch.arange(1, n + 1, dtype=ft.dtype, device=ft.device) / n - 1  # sample j in G_i: Exp((j/N - 1)·φ_i)
    turn = _matrix(_exp(frac[:, None] * phi[..., None, :]))  # (..., T, N, 3, 3)
    return _rotate_triplets(turn, ft).mean(dim=-2)


def impedance_one_step(state: Tensor, real: Tensor, ft: Tensor) -> tuple[Tensor, Tensor]:
    """Tier-3 one-step targets ``[local(x_k, x_{k+1}), W̄_{k+1} in G_k]`` (12) for ticks 0..T-2 of a (..., T) window of
    grip-site state8, and where x_{k+1} is known. W̄_i is the tick-mean wrench of frame i (`tick_mean_wrench`, over the
    tick ending at frame i). ``real`` marks frames inside the episode and ``ft`` holds each frame's (N, 6) F/T samples.
    The frame just past an episode's end is extrapolated (`_extend_past_end`) and its wrench holds the last frame's;
    x_{k+1} stays unknown in a window that starts at an episode's last frame."""
    state, known = _extend_past_end(state, real)
    state, ft = _promote(state, ft)
    with torch.autocast(device_type=state.device.type, enabled=False):
        pose = pose_from_state(state)
        rotation = _matrix(_unit(pose[..., 3:]))
        mean = tick_mean_wrench(state, ft)
        ahead = rotation[..., :-1, :, :].transpose(-1, -2) @ rotation[..., 1:, :, :]  # G_kᵀ G_{k+1}
        ahead = _rotate_triplets(ahead, mean[..., 1:, :])
        wrench = torch.where(~real[..., 1:, None], mean[..., :-1, :], ahead)  # k+1 past the end: W̄_k
        return torch.cat((local(pose[..., :-1, :], pose[..., 1:, :]), wrench), dim=-1), known[..., 1:]


def impedance_next_state(state: Tensor, step: Tensor) -> tuple[Tensor, Tensor]:
    """From a one-step target ``[local(x_k, x̂_{k+1}), W̄_{k+1} in G_k]`` (12) at state8 x_k: the state8 x̂_{k+1} (fingers
    held) and W̄_{k+1} in Ĝ_{k+1}, the drift's input for the model's next step."""
    state, step = _promote(state, step)
    with torch.autocast(device_type=state.device.type, enabled=False):
        pose = retract(pose_from_state(state), step[..., :6])
        turn = _matrix(_exp(_LIBERO_ANGULAR_PER_COMMAND * step[..., 3:6]))  # G_kᵀ Ĝ_{k+1}
        wrench = _rotate_triplets(turn.transpose(-1, -2), step[..., 6:12])
        return torch.cat((pose[..., :3], _log(pose[..., 3:]), state[..., 6:]), dim=-1), wrench


def impedance_action(first: Tensor, second: Tensor, gripper: Tensor) -> Tensor:
    """The impedance controller's action ``[local(x_k, x̂_{k+1}), local(x_k, x̂_{k+2}) in metres and radians,
    W̄_{k+1}, W̄_{k+2} in G_k, gripper]`` (25, envs/ft4_impedance.py) from the one-step targets ``first`` (12) at x_k
    and ``second`` (12) at x̂_{k+1}: local(x_k, x̂_{k+2}) composed, W̄_{k+2} turned into G_k."""
    first, second, gripper = _promote(first, second, gripper)
    with torch.autocast(device_type=first.device.type, enabled=False):
        origin = torch.zeros_like(first[..., :7])
        origin[..., 6] = 1  # x_k as the identity pose
        two = local(origin, retract(retract(origin, first[..., :6]), second[..., :6]))
        turn = _matrix(_exp(_LIBERO_ANGULAR_PER_COMMAND * first[..., 3:6]))  # G_kᵀ Ĝ_{k+1}
        units = first.new_tensor([_LIBERO_TRANSLATION_PER_COMMAND] * 3 + [_LIBERO_ANGULAR_PER_COMMAND] * 3).repeat(2)
        poses = torch.cat((first[..., :6], two), dim=-1) * units
        return torch.cat((poses, first[..., 6:12], _rotate_triplets(turn, second[..., 6:12]), gripper), dim=-1)


def pose_state_features(state: Tensor, reference_state: Tensor | None = None) -> Tensor:
    """Return ``[p, R[:,0], R[:,1], finger_qpos]``, optionally in a reference pose's frame.

    A broadcast-compatible reference gives ``p = R0.T @ (p - p0)`` and
    ``R = R0.T @ R``. Positions remain in meters and finger positions are unchanged.
    """
    if reference_state is None:
        (state,) = _promote(state)
    else:
        state, reference_state = _broadcast(*_promote(state, reference_state))
    with torch.autocast(device_type=state.device.type, enabled=False):
        matrix = _matrix(pose_from_state(state)[..., 3:])
        position = state[..., :3]
        if reference_state is not None:
            reference_inverse = _matrix(pose_from_state(reference_state)[..., 3:]).transpose(-1, -2)
            position = (reference_inverse @ (position - reference_state[..., :3])[..., None]).squeeze(-1)
            matrix = reference_inverse @ matrix
        return torch.cat((position, matrix[..., :, 0], matrix[..., :, 1], state[..., 6:]), dim=-1)


def action_to_endpoint(pose: Tensor, arm_action: Tensor) -> Tensor:
    """Decode a nominal pose7 from six WORLD controller commands.

    Commands are clipped component-wise to [-1, 1], then interpreted using the
    fixed LIBERO calibration: 0.05 m and 0.5 rad per command unit. Rotation is
    left-composed because the action is already in the world frame.
    """
    pose, arm_action = _broadcast(*_promote(pose, arm_action))
    with torch.autocast(device_type=pose.device.type, enabled=False):
        arm = arm_action.clamp(-1, 1)
        position = pose[..., :3] + _LIBERO_TRANSLATION_PER_COMMAND * arm[..., :3]
        quaternion = _multiply(_exp(_LIBERO_ANGULAR_PER_COMMAND * arm[..., 3:]), _unit(pose[..., 3:]))
        return torch.cat((position, quaternion), dim=-1)


def perturb_pose_state(state: Tensor, noise: Tensor) -> Tensor:
    """Retract normalized body-controller noise into raw state8, keeping fingers unchanged."""
    state, noise = _broadcast(*_promote(state, noise))
    with torch.autocast(device_type=state.device.type, enabled=False):
        pose = retract(pose_from_state(state), noise)
        return torch.cat((pose[..., :3], _log(pose[..., 3:]), state[..., 6:]), dim=-1)


def nearest_pose_indices(
    query: Tensor,
    candidates: Tensor,
    candidate_valid: Tensor,
) -> Tensor:
    """Return nearest valid candidate indices (B, H), rejecting empty chunks.

    Sanitize padded candidate payloads before geometry and mask costs before
    argmin. The caller handles query padding and must bypass nearest search when
    augmentation is off to preserve time alignment in duplicate-state ties.
    """
    if not bool(candidate_valid.any(dim=-1).all()):
        raise ValueError("Each chunk must have at least one valid candidate.")
    safe_candidates = torch.where(candidate_valid[..., None], candidates, torch.zeros_like(candidates))
    query, safe_candidates = _promote(query, safe_candidates)
    with torch.autocast(device_type=query.device.type, enabled=False):
        delta = local(pose_from_state(query)[:, :, None], pose_from_state(safe_candidates)[:, None])
        distances = delta.square().sum(dim=-1)
        finite = (torch.isfinite(distances) | ~candidate_valid[:, None]).all()
        if not bool(finite):
            raise ValueError("Each chunk needs finite valid-candidate distances.")
        return distances.masked_fill(~candidate_valid[:, None], torch.inf).argmin(dim=-1)
