#!/usr/bin/env python
"""FT-4 demo recorder, shared by generate_ft4.py and convert_native.py.

Frame t is the observation before action t:
- policy data: agentview and eye-in-hand RGB (256, rotated 180 deg like lerobot/libero), `observation.state`
  (8), `observation.ft_wrench` (25, 6) = EE-frame contact wrench of the control step that ended at t, `action` (7)
- depth of both cameras at t in metres (same rotation), one map per 20 Hz frame, stored as LeRobot depth video
- physics-rate (500 Hz) logs of the control step that executes action t, 26 time points (index 0 = frame start):
  the `hf.*` / `osc.*` keys of libero_hf/regenerate_libero_hf.py plus `hf.joint_pos` / `hf.joint_vel`
- `expert.phase`, the task's labels (`expert.*`, analysis only) and `meta.source` = (worker seed, episode
  index in that worker), which names the episode's side file

Side file <out>/side/w<seed>/ep<k>.npz: the episode's MJCF (zlib) and the flattened MuJoCo state at every frame start
plus the end state, so frames can be re-rendered later (env.reset_from_xml_string, sim.set_state_from_flattened).
Resets that write body placements into the compiled model (robosuite Door, the Wipe marker path) need those too:
convert_native.py stores them with its episodes (ep<k>_scene.npz).
Logging never changes the rollout: the extra `sim.forward()` calls leave the state and the warm start untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import multiprocessing as mp
import shutil
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_PORT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _PORT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


regen = load_module("regenerate_libero_hf", "libero_hf/regenerate_libero_hf.py")

N_SUB = 25
IMAGE_SIZE = 256
SETTLE_STEPS = 10
DEPTH = dict(depth_min=0.01, depth_max=3.0, shift=0.1, use_log=True)  # 12-bit log codes: 0.2 mm at 10 cm, 3 mm at 3 m


class SubstepLogger(regen.SubstepLogger):
    """regenerate_libero_hf.SubstepLogger with joint positions/velocities, and zeros for finger-less grippers."""

    def __init__(self, robo_env):
        super().__init__(robo_env)
        self.q_idx = list(self.robot._ref_joint_pos_indexes)
        self.v_idx = list(self.robot._ref_joint_vel_indexes)

    def _sample(self) -> np.ndarray:
        c, d = self.ctrl, self.d
        r_ft = np.array(d.site_xmat[self.ft_site]).reshape(3, 3)
        f_s, t_s = np.array(self.robot.ee_force), np.array(self.robot.ee_torque)
        f_w, t_w = r_ft @ f_s, r_ft @ t_s
        p_ee = np.array(c.ee_pos)
        w_contact, arm = self._contact_wrench(p_ee)
        grip = d.qpos[self.grip_q_idx] if len(self.grip_q_idx) else np.zeros(2)
        return np.concatenate([
            p_ee, regen._mat2quat_xyzw(c.ee_ori_mat), c.ee_pos_vel, c.ee_ori_vel,  # 0:13
            f_w, t_w, f_s, t_s, grip, d.ctrl[self.act_idx], w_contact,  # 13:40
            self._ft_wrench(p_ee, f_w, t_w), [arm],  # 40:47
            d.qpos[self.q_idx], d.qvel[self.v_idx],  # 47:61
        ])

    def frame_end(self) -> dict:
        out = super().frame_end()
        s = np.stack(self.samples)
        out["hf.joint_pos"], out["hf.joint_vel"] = s[:, 47:54], s[:, 54:61]
        return out


def features(hidden: dict) -> dict:
    """Dataset features; `hidden` maps expert.* label names to feature specs."""
    image = {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channels"]}
    depth = {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 1), "names": ["height", "width", "channels"],
             "info": {"is_depth_map": True}}
    hf = {k: v for k, v in regen.hf_features(N_SUB).items() if k != "raw.demo_index"}
    joints = {"dtype": "float32", "shape": (N_SUB + 1, 7), "names": None}
    return {
        "observation.images.image": image,
        "observation.images.image2": image,
        "observation.images.image_depth": depth,
        "observation.images.image2_depth": depth,
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "expert.phase": {"dtype": "int64", "shape": (1,), "names": None},
        **hidden,
        "meta.source": {"dtype": "int64", "shape": (2,), "names": None},
        **hf,  # includes observation.ft_wrench (25, 6)
        "hf.joint_pos": joints,
        "hf.joint_vel": joints,
    }


def state8(obs: dict, fingers: bool = True) -> np.ndarray:
    from robosuite.utils.transform_utils import quat2axisangle

    grip = obs["robot0_gripper_qpos"] if fingers else np.zeros(2)
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), grip])


def rollout(env, make_expert, rng, *, settle_action, max_steps, act_dims=7, fingers=True, record=True, on_step=None):
    """Expert rollout from reset. Frame t: obs(t), F/T block seen at t, action(t).

    Returns (frames, info); with `record`, frames carry images, depth and 500 Hz logs, and info the side-file data.
    The expert sees the FTWrenchLogger block of the last control step.
    """
    from robosuite.utils.camera_utils import get_real_depth_map

    from lerobot.envs.libero_ft import FTWrenchLogger

    env.reset()
    ft = FTWrenchLogger(env)
    log = SubstepLogger(env) if record else None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(settle_action)
    expert = make_expert(env, rng)
    xml = env.model.get_xml() if record else None
    frames, states, success = [], [], False
    for _ in range(max_steps):
        block = ft.block()
        action = expert.act(block)
        fr = {"observation.state": state8(obs, fingers), "observation.ft_wrench": block, "action": action,
              "phase": expert.phase}
        if record:
            for key, cam in (("image", "agentview"), ("image2", "robot0_eye_in_hand")):
                fr[f"observation.images.{key}"] = obs[f"{cam}_image"][::-1, ::-1].copy()
                fr[f"observation.images.{key}_depth"] = (
                    get_real_depth_map(env.sim, obs[f"{cam}_depth"])[::-1, ::-1].astype(np.float16))
            fr.update(log.frame_start())
            states.append(env.sim.get_state().flatten())
        obs, _, _, _ = env.step(action[:act_dims])
        if record:
            fr.update(log.frame_end())
        frames.append(fr)
        if on_step is not None:
            on_step(env, expert)
        if env._check_success():
            success = True
            break
        if getattr(env, "failed", False) or getattr(env, "overforce", False) or expert.failed:
            break
    if record:
        states.append(env.sim.get_state().flatten())
        log.detach()
    ft.detach()
    info = {"success": success, "steps": len(frames), "expert": expert,
            "phases": np.bincount([fr["phase"] for fr in frames], minlength=16)}
    if record:
        info.update(xml=xml, states=np.stack(states))
    return frames, info


class ShardWriter:
    """One LeRobot dataset per worker under <out>/shards/w<seed>, side files under <out>/side/w<seed>."""

    def __init__(self, out: str, seed: int, hidden: dict):
        from lerobot.configs import DepthEncoderConfig
        from lerobot.datasets import LeRobotDataset

        self.seed, self.k = seed, 0
        self.side = Path(out) / "side" / f"w{seed}"
        self.side.mkdir(parents=True, exist_ok=True)
        self.ds = LeRobotDataset.create(
            repo_id=f"local/w{seed}", fps=20, features=features(hidden), root=Path(out) / "shards" / f"w{seed}",
            robot_type="panda", use_videos=True, image_writer_threads=4,
            depth_encoder=DepthEncoderConfig(
                **DEPTH, preset="ultrafast", extra_options={"x265-params": "lossless=1:log-level=error"}
            ),
        )

    def add(self, frames: list, info: dict, task: str, labels: dict) -> None:
        """Save one episode; `labels` are its expert.* values (arrays)."""
        for fr in frames:
            row = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype not in (np.uint8,) else v)
                   for k, v in fr.items() if k != "phase"}
            row["expert.phase"] = np.array([fr["phase"]], dtype=np.int64)
            row["meta.source"] = np.array([self.seed, self.k], dtype=np.int64)
            row.update(labels)
            row["task"] = task
            self.ds.add_frame(row)
        self.ds.save_episode()
        np.savez_compressed(self.side / f"ep{self.k:04d}.npz", states=info["states"],
                            xml=np.frombuffer(zlib.compress(info["xml"].encode()), dtype=np.uint8),
                            **{k: np.asarray(v) for k, v in labels.items()})
        self.k += 1

    def close(self) -> None:
        self.ds.finalize()


def run_shards(worker, jobs: list, workers: int) -> list:
    """Run `worker(job)` over jobs in spawn processes (mp.Pool workers cannot start the dataset writer's processes)."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(worker, jobs))


def split(episodes: int, parts: int) -> list[int]:
    return [episodes // parts + (i < episodes % parts) for i in range(parts)]


def aggregate(out: str, repo_id: str) -> None:
    """<out>/final from the shards (Array2D-safe aggregate of regenerate_libero_hf)."""
    regen.cmd_aggregate(argparse.Namespace(out_dir=out, repo_id=repo_id))


def slim(src: Path, dst: Path, drop: tuple[str, ...]) -> None:
    """View of `src` without the features whose names start with `drop`; videos are symlinked."""
    import pyarrow.parquet as pq

    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode

    gone = lambda name: name.startswith(drop)  # noqa: E731
    shutil.rmtree(dst, ignore_errors=True)
    for f in sorted((src / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        table = table.select([c for c in table.column_names if not gone(c)])
        hf = json.loads(table.schema.metadata[b"huggingface"])
        hf["info"]["features"] = {k: v for k, v in hf["info"]["features"].items() if not gone(k)}
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_table_one_row_group_per_episode(table.replace_schema_metadata({b"huggingface": json.dumps(hf)}), out)
    for f in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(f)
        keep = [c for c in table.column_names
                if not (c.startswith(("stats/", "videos/")) and gone(c.split("/")[1]))]
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.select(keep).replace_schema_metadata(None), out)
    info = json.loads((src / "meta" / "info.json").read_text())
    info["features"] = {k: v for k, v in info["features"].items() if not gone(k)}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    stats = json.loads((src / "meta" / "stats.json").read_text())
    (dst / "meta" / "stats.json").write_text(json.dumps({k: v for k, v in stats.items() if not gone(k)}, indent=4))
    shutil.copy(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")
    (dst / "videos").symlink_to(src / "videos")


def views(out: str) -> None:
    """<out>/policy (RGB, state, F/T, action, expert.phase) and <out>/policy_depth (+ depth) next to <out>/final."""
    final = Path(out) / "final"
    hidden = tuple(k for k in json.loads((final / "meta" / "info.json").read_text())["features"]
                   if k.startswith("expert.") and k != "expert.phase")
    base = ("hf.", "osc.", "meta.") + hidden
    slim(final, Path(out) / "policy", base + ("observation.images.image_depth", "observation.images.image2_depth"))
    slim(final, Path(out) / "policy_depth", base)
    logging.info("views: %s/{policy,policy_depth}", out)


def report(res: list, t0: float) -> None:
    ok, tried = map(sum, zip(*res))
    print(f"{ok} episodes kept of {tried} in {time.time() - t0:.0f} s (expert success {ok / max(tried, 1):.3f})",
          flush=True)
