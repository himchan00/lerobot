#!/usr/bin/env python
"""Regenerate LIBERO demos with physics-rate (500 Hz) EE state, wrist F/T and OSC internals.

`lerobot/libero` stores only 8-D state, 7-D action and two cameras per control step, and no MuJoCo
state, so it cannot be replayed. This script replays the raw LIBERO hdf5 demos (which store
`states` and `actions`) with the same procedure OpenVLA used to build the data behind
`lerobot/libero` (10 settle steps, no-op filtering, 256x256 render, keep successful replays,
images rotated 180 deg) and adds a block of physics substeps to every frame.

Frame t holds the observation before action t (as in `lerobot/libero`). Its `hf.*` block holds
N+1 = 26 time points of the control step that executes action t (20 Hz control, 2 ms physics):
index 0 is the frame start, the state the OSC reads when it sets the goal; index k is the state
after k physics steps, so `hf.*[t, -1] == hf.*[t+1, 0]`. robosuite samples observables one physics
step early, so `observation.state` of frame t+1 equals `hf.*[t, -2]` (<1 mm from `hf.*[t+1, 0]`).
That holds with the logger attached: its per-step refresh makes observables read the post-step state.
The stock env (and `lerobot/libero`) samples one step earlier, `hf.*[t, -3]` (<0.5 mm apart); the
rollout itself is identical. LIBERO eval with `--env.ft_wrench=true` logs the same way.

`hf.*` / `osc.*` keys never start with `observation` or `action`, so `dataset_to_policy_features`
ignores them; the policy preprocessor also drops them from training batches. The one policy-facing
extra is `observation.ft_wrench` (N, 6): the environment-interaction wrench at the wrist over the
control step that ENDED at frame t (time points 1..N of the previous block; frame 0 gets the last
settle step), i.e. `hf.contact_wrench_ee` (what a gravity- and inertia-compensated wrist sensor
reads), rotated into the EE frame per time point, [F, tau] in N and N m. The raw reading minus the
static weight (`hf.ft_wrench_ee`) also carries the gripper's inertial reaction to the last command,
which a policy learns to copy. `lerobot.envs.libero_ft` computes the same during evaluation.

Usage (one process per task, then aggregate):

    python regenerate_libero_hf.py download --raw-dir /PublicSSD/himchan/libero_raw
    python regenerate_libero_hf.py check --raw-dir ... --suite libero_spatial
    python regenerate_libero_hf.py replay --raw-dir ... --out-dir ... --suites libero_spatial --num-workers 10
    python regenerate_libero_hf.py aggregate --out-dir ... --repo-id himchan00/libero_hf
    python regenerate_libero_hf.py slim --out-dir ...   # policy view without hf.*/osc.*/raw.*
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
RAW_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
IMAGE_RESOLUTION = 256
NUM_SETTLE_STEPS = 10  # OpenVLA regeneration and lerobot LiberoEnv.num_steps_wait
FPS_LABEL = 10  # lerobot/libero label; one frame is one 20 Hz control step (see module docstring)
N_SUB = 25


# ----------------------------------------------------------------------------------------------
# Substep logging (robosuite 1.4, pinned by hf-libero)
# ----------------------------------------------------------------------------------------------


def _mat2quat_xyzw(mat: np.ndarray) -> np.ndarray:
    import robosuite.utils.transform_utils as T

    return T.mat2quat(mat)


class SubstepLogger:
    """Wraps `sim.step` of a robosuite env and records the state at every physics step.

    A control step yields N+1 time points (N = 25 substeps): index 0 is the frame start (after
    `mj_forward`, the state the OSC reads when it sets the goal), index k is the state after k
    `mj_step`s. Each sample is taken after `mj_forward` (forced controller update), so pose,
    velocity, sensors, contacts and kinematics refer to the same time. MuJoCo does not update
    `qacc_warmstart` in `mj_forward` and the controller's `new_update` flag is restored, so logging
    does not change the rollout (`check` verifies this on a real LIBERO demo).

    Wrenches are world-frame [force, torque] about the EE point (OSC `grip_site`):
      - contact wrench: ground truth, sum of MuJoCo contact forces on the bodies below the F/T sensor.
      - ft wrench: what a real F/T pipeline computes, i.e. the sensor reading with the static weight
        of the bodies below the sensor removed and moved to the EE point (inertial terms remain).
    """

    G = np.array([0.0, 0.0, -9.81])

    def __init__(self, robo_env):
        import mujoco

        self._mj = mujoco
        self.env = robo_env
        self.robot = robo_env.robots[0]
        self.ctrl = self.robot.controller
        sim = robo_env.sim
        self.m, self.d = sim.model._model, sim.data._data
        prefix = self.robot.gripper.naming_prefix
        self.ft_site = sim.model.site_name2id(f"{prefix}ft_frame")
        self.ee_site = sim.model.site_name2id(self.ctrl.eef_name)
        self.ft_body = int(self.m.site_bodyid[self.ft_site])
        self.below_ft = self._subtree(self.ft_body)
        robot_root = sim.model.body_name2id(self.robot.robot_model.root_body)
        self.arm_bodies = self._subtree(robot_root) - self.below_ft
        self.mass_below_ft = float(self.m.body_subtreemass[self.ft_body])
        self.grip_q_idx = list(self.robot._ref_gripper_joint_pos_indexes)
        self.act_idx = list(self.robot._ref_joint_actuator_indexes)
        self.n_sub = int(round(robo_env.control_timestep / robo_env.model_timestep))
        self._f6 = np.zeros(6)
        self._orig_step = sim.step
        sim.step = self._step  # instance override; robosuite calls self.sim.step()
        self.samples: list[np.ndarray] = []

    def _subtree(self, root: int) -> set[int]:
        out = set()
        for b in range(self.m.nbody):
            p = b
            while p != 0 and p != root:
                p = int(self.m.body_parentid[p])
            if p == root:
                out.add(b)
        return out

    def detach(self):
        self.env.sim.step = self._orig_step

    def _refresh(self):
        flag = self.ctrl.new_update
        self.ctrl.update(force=True)  # runs sim.forward()
        self.ctrl.new_update = flag

    def _contact_wrench(self, p_ee: np.ndarray) -> tuple[np.ndarray, float]:
        """Contact wrench on the bodies below the F/T sensor about p_ee (world), and |F| on other arm links."""
        m, d = self.m, self.d
        w, arm = np.zeros(6), 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            in1, in2 = b1 in self.below_ft, b2 in self.below_ft
            if in1 == in2:  # internal to the gripper, or not touching it
                if (b1 in self.arm_bodies) != (b2 in self.arm_bodies):
                    self._mj.mj_contactForce(m, d, i, self._f6)
                    arm += float(np.linalg.norm(self._f6[:3]))
                continue
            self._mj.mj_contactForce(m, d, i, self._f6)
            frame = np.asarray(c.frame).reshape(3, 3)  # rows: normal (geom1 -> geom2), tangents
            sign = -1.0 if in1 else 1.0  # mj_contactForce gives the force on geom2
            force = sign * (frame.T @ self._f6[:3])
            torque = sign * (frame.T @ self._f6[3:]) + np.cross(np.asarray(c.pos) - p_ee, force)
            w[:3] += force
            w[3:] += torque
        return w, arm

    def _ft_wrench(self, p_ee: np.ndarray, f_world: np.ndarray, t_world: np.ndarray) -> np.ndarray:
        """Gravity-compensated F/T reading as an external wrench about p_ee (world)."""
        d = self.d
        p_ft = np.array(d.site_xpos[self.ft_site])
        com = np.array(d.subtree_com[self.ft_body])
        w_grav_force = self.mass_below_ft * self.G
        w_grav_torque = np.cross(com - p_ft, w_grav_force)
        force = -f_world - w_grav_force  # sensor reads the reaction of (contacts + weight) below it
        torque_ft = -t_world - w_grav_torque
        return np.concatenate([force, torque_ft + np.cross(p_ft - p_ee, force)])

    def _sample(self) -> np.ndarray:
        c, d = self.ctrl, self.d
        r_ft = np.array(d.site_xmat[self.ft_site]).reshape(3, 3)
        f_s, t_s = np.array(self.robot.ee_force), np.array(self.robot.ee_torque)
        f_w, t_w = r_ft @ f_s, r_ft @ t_s
        p_ee = np.array(c.ee_pos)
        w_contact, arm = self._contact_wrench(p_ee)
        return np.concatenate(
            [
                p_ee,  # 0:3
                _mat2quat_xyzw(c.ee_ori_mat),  # 3:7
                c.ee_pos_vel,  # 7:10
                c.ee_ori_vel,  # 10:13
                f_w,  # 13:16 raw F/T force, world axes
                t_w,  # 16:19 raw F/T torque, world axes, about the sensor site
                f_s,  # 19:22 raw F/T force, sensor frame
                t_s,  # 22:25 raw F/T torque, sensor frame
                d.qpos[self.grip_q_idx],  # 25:27
                d.ctrl[self.act_idx],  # 27:34 joint torque command applied in the next mj_step
                w_contact,  # 34:40 ground-truth contact wrench below the sensor, about the EE point
                self._ft_wrench(p_ee, f_w, t_w),  # 40:46 gravity-compensated F/T wrench about the EE point
                [arm],  # 46 contact force magnitude on other arm links (model invalid if > 0)
            ]
        )

    def _step(self, *args, **kwargs):
        self._orig_step(*args, **kwargs)
        self._refresh()
        self.samples.append(self._sample())

    def frame_start(self) -> dict:
        """Index-0 sample and OSC task-space inertia at the start of a control step."""
        from robosuite.utils.control_utils import opspace_matrices

        self._refresh()
        c = self.ctrl
        lam_full, lam_pos, lam_ori, _ = opspace_matrices(c.mass_matrix, c.J_full, c.J_pos, c.J_ori)
        self.samples = [self._sample()]
        return {"osc.lambda_full": lam_full, "osc.lambda_pos": lam_pos, "osc.lambda_ori": lam_ori}

    def frame_end(self) -> dict:
        s = np.stack(self.samples)
        if s.shape[0] != self.n_sub + 1:
            raise RuntimeError(f"expected {self.n_sub + 1} time points, got {s.shape[0]}")
        c = self.ctrl
        return {
            "hf.ee_pos": s[:, 0:3],
            "hf.ee_quat": s[:, 3:7],
            "hf.ee_linvel": s[:, 7:10],
            "hf.ee_angvel": s[:, 10:13],
            "hf.ft_force_world": s[:, 13:16],
            "hf.ft_torque_world": s[:, 16:19],
            "hf.ft_force_sensor": s[:, 19:22],
            "hf.ft_torque_sensor": s[:, 22:25],
            "hf.gripper_qpos": s[:, 25:27],
            "hf.joint_torque_cmd": s[:, 27:34],
            "hf.contact_wrench_ee": s[:, 34:40],
            "hf.ft_wrench_ee": s[:, 40:46],
            "hf.arm_contact_force": s[:, 46:47],
            "osc.goal_pos": np.array(c.goal_pos),
            "osc.goal_quat": _mat2quat_xyzw(np.array(c.goal_ori)),
            "osc.kp": np.array(c.kp, dtype=np.float64) * np.ones(6),
            "osc.kd": np.array(c.kd, dtype=np.float64) * np.ones(6),
        }


def hf_features(n_sub: int = N_SUB) -> dict:
    n = n_sub + 1

    def f(shape, names=None):
        return {"dtype": "float32", "shape": tuple(shape), "names": names}

    return {
        "hf.ee_pos": f((n, 3)),
        "hf.ee_quat": f((n, 4)),
        "hf.ee_linvel": f((n, 3)),
        "hf.ee_angvel": f((n, 3)),
        "hf.ft_force_world": f((n, 3)),
        "hf.ft_torque_world": f((n, 3)),
        "hf.ft_force_sensor": f((n, 3)),
        "hf.ft_torque_sensor": f((n, 3)),
        "hf.gripper_qpos": f((n, 2)),
        "hf.joint_torque_cmd": f((n, 7)),
        "hf.contact_wrench_ee": f((n, 6)),
        "hf.ft_wrench_ee": f((n, 6)),
        "hf.arm_contact_force": f((n, 1)),
        "osc.goal_pos": f((3,)),
        "osc.goal_quat": f((4,)),
        "osc.kp": f((6,)),
        "osc.kd": f((6,)),
        "osc.lambda_full": f((6, 6)),
        "osc.lambda_pos": f((3, 3)),
        "osc.lambda_ori": f((3, 3)),
        "raw.demo_index": {"dtype": "int64", "shape": (1,), "names": None},
        "observation.ft_wrench": f((n_sub, 6)),  # policy input, see module docstring
    }


LIBERO_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}


# ----------------------------------------------------------------------------------------------
# Replay (OpenVLA procedure + logging)
# ----------------------------------------------------------------------------------------------


def is_noop(action, prev_action=None, threshold=1e-4):
    """OpenVLA `regenerate_libero_dataset.is_noop`."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def state8(obs) -> np.ndarray:
    import robosuite.utils.transform_utils as T

    return np.concatenate(
        [obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    )


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def ft_wrench_body(block: dict) -> np.ndarray:
    """`observation.ft_wrench` of the next frame: `hf.contact_wrench_ee` at time points 1..N, EE frame."""
    rt = np.stack([_quat_xyzw_to_mat(q).T for q in block["hf.ee_quat"][1:]])
    w = block["hf.contact_wrench_ee"][1:]
    return np.concatenate([(rt @ w[:, :3, None])[..., 0], (rt @ w[:, 3:, None])[..., 0]], axis=-1)


def replay_demo(env, robo_env, init_state, actions, dummy_action, image_keys=None):
    """Replay one demo. Returns (frames, sim_states, info). `env` is the LIBERO ControlEnv
    (or any object with reset/set_init_state/step), `robo_env` the underlying robosuite env."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(NUM_SETTLE_STEPS - 1):
        obs, _, _, _ = env.step(dummy_action)

    logger = SubstepLogger(robo_env)
    frames, sim_states, kept, n_noop, done = [], [], [], 0, False
    try:
        # The last settle step is logged too: its F/T is frame 0's `observation.ft_wrench`.
        logger.frame_start()
        obs, _, _, _ = env.step(dummy_action)
        ft_wrench = ft_wrench_body(logger.frame_end())
        for action in actions:
            prev = kept[-1] if kept else None
            if is_noop(action, prev):
                n_noop += 1
                continue
            frame = {"observation.state": state8(obs), "observation.ft_wrench": ft_wrench, "action": np.asarray(action)}
            if image_keys is not None:
                # OpenVLA RLDS / lerobot/libero store images rotated by 180 deg
                frame["observation.images.image"] = obs[image_keys[0]][::-1, ::-1]
                frame["observation.images.image2"] = obs[image_keys[1]][::-1, ::-1]
            sim_states.append(robo_env.sim.get_state().flatten())
            frame.update(logger.frame_start())
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            frame.update(logger.frame_end())
            ft_wrench = ft_wrench_body(frame)
            frames.append(frame)
            kept.append(action)
    finally:
        logger.detach()
    info = {"success": bool(done), "n_frames": len(frames), "n_noops": n_noop}
    return frames, np.stack(sim_states) if sim_states else None, info


# ----------------------------------------------------------------------------------------------
# Per-task worker
# ----------------------------------------------------------------------------------------------


def _shard_name(suite: str, task_id: int) -> str:
    return f"{suite}_task{task_id:02d}"


def replay_task(raw_dir: str, out_dir: str, suite: str, task_id: int, max_demos: int | None) -> dict:
    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from lerobot.datasets import LeRobotDataset

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=IMAGE_RESOLUTION, camera_widths=IMAGE_RESOLUTION
    )
    env.seed(0)

    raw_path = Path(raw_dir) / suite / f"{task.name}_demo.hdf5"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    out = Path(out_dir)
    shard = _shard_name(suite, task_id)
    shard_root = out / "shards" / shard
    side_dir = out / "sim_states" / suite / task.name
    side_dir.mkdir(parents=True, exist_ok=True)

    features = {**LIBERO_FEATURES, **hf_features(N_SUB)}
    ds = LeRobotDataset.create(
        repo_id=f"local/{shard}",
        fps=FPS_LABEL,
        features=features,
        root=shard_root,
        robot_type="panda",
        use_videos=True,
        image_writer_threads=4,
    )

    log = {"suite": suite, "task_id": task_id, "task": task.language, "demos": {}}
    t0 = time.time()
    with h5py.File(raw_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        for key in demo_keys:
            i = int(key.split("_")[1])
            g = f["data"][key]
            frames, states, info = replay_demo(
                env,
                env.env,
                g["states"][0],
                g["actions"][()],
                [0, 0, 0, 0, 0, 0, -1],
                image_keys=("agentview_image", "robot0_eye_in_hand_image"),
            )
            log["demos"][key] = info
            if not info["success"]:
                continue
            for fr in frames:
                out_fr = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype != np.uint8 else v)
                          for k, v in fr.items()}
                out_fr["raw.demo_index"] = np.array([i], dtype=np.int64)
                out_fr["task"] = task.language
                ds.add_frame(out_fr)
            ds.save_episode()
            np.savez_compressed(side_dir / f"{key}.npz", states=states)
    ds.finalize()
    gripper_mass = _gripper_mass(env)
    env.close()

    n = len(log["demos"])
    n_ok = sum(d["success"] for d in log["demos"].values())
    log.update(
        {
            "n_replayed": n,
            "n_success": n_ok,
            "seconds": time.time() - t0,
            "gripper_subtree_mass": gripper_mass,
        }
    )
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs" / f"{shard}.json").write_text(json.dumps(log, indent=2))
    return {"shard": shard, "n_replayed": n, "n_success": n_ok, "seconds": log["seconds"]}


def _gripper_mass(env) -> float | None:
    """Mass below the F/T site, for gravity compensation of `hf.ft_*`."""
    try:
        robo_env = env.env
        sim = robo_env.sim
        prefix = robo_env.robots[0].gripper.naming_prefix
        body = sim.model.site_bodyid[sim.model.site_name2id(f"{prefix}ft_frame")]
        return float(sim.model.body_subtreemass[body])
    except Exception:  # noqa: BLE001 - diagnostic only
        return None


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def cmd_download(args):
    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(RAW_REPO_ID, repo_type="dataset")
    folders = sorted({p.split("/")[0] for p in files if "/" in p})
    print("raw repo folders:", folders)
    patterns = [f"{s}/*" for s in args.suites if s in folders]
    missing = [s for s in args.suites if s not in folders]
    if missing:
        print("not found as folders (check the list above):", missing)
    snapshot_download(RAW_REPO_ID, repo_type="dataset", local_dir=args.raw_dir, allow_patterns=patterns)


def cmd_replay(args):
    from libero.libero import benchmark

    jobs = []
    for suite in args.suites:
        n_tasks = benchmark.get_benchmark_dict()[suite]().n_tasks
        ids = args.task_ids if args.task_ids is not None else range(n_tasks)
        jobs += [(suite, t) for t in ids]
    done_dir = Path(args.out_dir) / "logs"
    if args.skip_done:
        jobs = [j for j in jobs if not (done_dir / f"{_shard_name(*j)}.json").exists()]
    logging.info("%d task jobs", len(jobs))
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futs = {
            pool.submit(replay_task, args.raw_dir, args.out_dir, s, t, args.max_demos): (s, t) for s, t in jobs
        }
        for fut in as_completed(futs):
            s, t = futs[fut]
            try:
                r = fut.result()
                logging.info("done %s: %d/%d success, %.0fs", r["shard"], r["n_success"], r["n_replayed"], r["seconds"])
            except Exception as e:  # noqa: BLE001
                logging.exception("FAILED %s task %d: %s", s, t, e)


def cmd_aggregate(args):
    import datasets
    import pandas as pd
    from datasets.features.features import PandasArrayExtensionDtype

    import lerobot.datasets.aggregate as aggregate_module
    from lerobot.datasets import LeRobotDatasetMetadata, aggregate_datasets
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode

    shard_dir = Path(args.out_dir) / "shards"
    shards = sorted(p for p in shard_dir.iterdir() if (p / "meta" / "info.json").exists())
    logging.info("aggregating %d shards", len(shards))

    # lerobot merges data parquet through pandas. The 2-D (Array2D) columns read back as datasets'
    # PandasArrayExtensionDtype, which pandas cannot concat and pyarrow cannot write, so read them as
    # plain object columns and write through HF datasets with the shard features.
    hf_features = get_hf_features_from_features(LeRobotDatasetMetadata("local/shard", root=shards[0]).features)

    def read_parquet(path, *a, **kw):
        df = pd.read_parquet(path, *a, **kw)
        for col in df.columns:
            if isinstance(df[col].dtype, PandasArrayExtensionDtype):
                df[col] = pd.Series(list(np.asarray(df[col].array)), index=df.index, dtype=object)
        return df

    def to_parquet(df, path):
        table = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=hf_features).with_format("arrow")[:]
        write_table_one_row_group_per_episode(table, path)

    class _Pandas:
        def __getattr__(self, name):
            return read_parquet if name == "read_parquet" else getattr(pd, name)

    aggregate_module.pd = _Pandas()
    aggregate_module.to_parquet_one_row_group_per_episode = to_parquet
    aggregate_datasets(
        repo_ids=[f"local/{p.name}" for p in shards],
        aggr_repo_id=args.repo_id,
        roots=shards,
        aggr_root=Path(args.out_dir) / "final",
    )


def cmd_slim(args):
    """`<out-dir>/policy`: `final` without the `hf.*`, `osc.*` and `raw.*` columns, videos linked.

    lerobot gathers whole rows for every multi-frame window, so those wide columns make policy
    training read ~10x slower; ODE work reads them from `final`.
    """
    import shutil

    import pyarrow.parquet as pq

    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode

    drop = ("hf.", "osc.", "raw.")
    src, dst = Path(args.out_dir) / "final", Path(args.out_dir) / "policy"
    shutil.rmtree(dst, ignore_errors=True)
    for f in sorted((src / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        table = table.select([c for c in table.column_names if not c.startswith(drop)])
        hf = json.loads(table.schema.metadata[b"huggingface"])
        hf["info"]["features"] = {k: v for k, v in hf["info"]["features"].items() if not k.startswith(drop)}
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_table_one_row_group_per_episode(table.replace_schema_metadata({b"huggingface": json.dumps(hf)}), out)
    for f in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(f)
        table = table.select(
            [c for c in table.column_names if not (c.startswith("stats/") and c.split("/")[1].startswith(drop))]
        )
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(None), out)
    info = json.loads((src / "meta" / "info.json").read_text())
    info["features"] = {k: v for k, v in info["features"].items() if not k.startswith(drop)}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    stats = json.loads((src / "meta" / "stats.json").read_text())
    (dst / "meta" / "stats.json").write_text(json.dumps({k: v for k, v in stats.items() if not k.startswith(drop)}, indent=4))
    shutil.copy(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    (dst / "videos").symlink_to(src / "videos")
    logging.info("policy view: %s (%d features)", dst, len(info["features"]))


def cmd_check(args):
    """Replay one raw demo with and without logging and report the invariants."""
    import h5py
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=IMAGE_RESOLUTION, camera_widths=IMAGE_RESOLUTION)
    env.seed(0)
    dummy = [0, 0, 0, 0, 0, 0, -1]
    with h5py.File(Path(args.raw_dir) / args.suite / f"{task.name}_demo.hdf5", "r") as f:
        g = f["data"][f"demo_{args.demo}"]
        init, actions = g["states"][0], g["actions"][()]
        print("raw demo keys:", list(g.keys()), "| obs keys:", list(g["obs"].keys()) if "obs" in g else None)

    t0 = time.time()
    frames, _, info = replay_demo(env, env.env, init, actions, dummy, image_keys=None)
    t_log = time.time() - t0
    s_logged = env.env.sim.get_state().flatten()

    env.reset()
    env.set_init_state(init)
    for _ in range(NUM_SETTLE_STEPS):
        env.step(dummy)
    for fr in frames:
        env.step(fr["action"].tolist())
    s_plain = env.env.sim.get_state().flatten()

    p_next = np.stack([fr["observation.state"][:3] for fr in frames[1:]])
    hf = np.stack([fr["hf.ee_pos"] for fr in frames[:-1]])
    hf_next0 = np.stack([fr["hf.ee_pos"][0] for fr in frames[1:]])
    goal = np.stack([fr["osc.goal_pos"] for fr in frames])
    p0 = np.stack([fr["hf.ee_pos"][0] for fr in frames])
    a = np.clip(np.stack([fr["action"][:3] for fr in frames]), -1, 1)
    w_c = np.concatenate([fr["hf.contact_wrench_ee"] for fr in frames])
    w_f = np.concatenate([fr["hf.ft_wrench_ee"] for fr in frames])
    arm = np.concatenate([fr["hf.arm_contact_force"] for fr in frames])
    print(json.dumps(info))
    print(f"task: {task.language} | replay time {t_log:.1f}s for {len(frames)} frames")
    print(f"logging changes rollout: max|state diff| = {np.abs(s_logged - s_plain).max():.2e}")
    print(f"hf[t,-1] vs hf[t+1,0]: {np.abs(hf[:, -1] - hf_next0).max():.2e} m (must be 0)")
    print(f"obs state[t+1] vs hf[t,-2]: {np.abs(hf[:, -2] - p_next).max():.2e} m | vs hf[t,-1]: {np.abs(hf[:, -1] - p_next).max():.2e} m")
    print(f"osc goal vs hf[t,0] + 0.05 a (world): {np.abs(goal - (p0 + 0.05 * a)).max():.2e} m")
    print(f"kp {frames[0]['osc.kp']} | kd {frames[0]['osc.kd']}")
    fc = np.linalg.norm(w_c[:, :3], axis=-1)
    print(f"GT contact |F| on gripper (N) 50/90/99/max: {np.percentile(fc, [50, 90, 99, 100]).round(2)}")
    print(f"F/T-derived minus GT force (N) RMS / max: {np.sqrt(((w_f[:, :3] - w_c[:, :3]) ** 2).mean()):.3f} / {np.abs(w_f[:, :3] - w_c[:, :3]).max():.3f}")
    print(f"time points with contact on other arm links: {(arm[:, 0] > 1e-6).mean() * 100:.2f} % (max {arm.max():.2f} N)")
    print(f"mass below F/T sensor: {_gripper_mass(env)} kg")
    env.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--raw-dir", required=True)
    d.add_argument("--suites", nargs="+", default=list(SUITES))

    c = sub.add_parser("check")
    c.add_argument("--raw-dir", required=True)
    c.add_argument("--suite", default="libero_spatial")
    c.add_argument("--task-id", type=int, default=0)
    c.add_argument("--demo", type=int, default=0)

    r = sub.add_parser("replay")
    r.add_argument("--raw-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--suites", nargs="+", default=list(SUITES))
    r.add_argument("--task-ids", nargs="+", type=int, default=None)
    r.add_argument("--max-demos", type=int, default=None, help="smoke test: first N demos per task")
    r.add_argument("--num-workers", type=int, default=8)
    r.add_argument("--skip-done", action="store_true")

    a = sub.add_parser("aggregate")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--repo-id", default="himchan00/libero_hf")

    s = sub.add_parser("slim")
    s.add_argument("--out-dir", required=True)

    args = p.parse_args()
    {"download": cmd_download, "check": cmd_check, "replay": cmd_replay, "aggregate": cmd_aggregate, "slim": cmd_slim}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
