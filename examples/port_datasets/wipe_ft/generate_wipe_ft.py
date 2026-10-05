#!/usr/bin/env python
"""Wipe-FT demos from a scripted force-regulating expert (task: lerobot/envs/wipe_ft.py).

The expert knows the marker path, never the hidden surface pose. It lowers the tool over the first marker until the
wrist F/T reads contact, then follows the path while setting its height target every step from the measured normal
force (admittance toward 10 N), so the vertical command is a per-step function of the current F/T.

usage (latent_sde env):
  MUJOCO_GL=egl python generate_wipe_ft.py pilot --episodes 100 --workers 20 --out pilot.npz
  MUJOCO_GL=osmesa LP_NUM_THREADS=1 python generate_wipe_ft.py shards --episodes 1000 --workers 16 --seed 600000 --out <dir>
  python generate_wipe_ft.py aggregate --out <dir>
Shared pieces (dataset features, aggregation, state8) come from ../square_ft/generate_square_ft.py.
"""

import argparse
import importlib.util
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "generate_square_ft", Path(__file__).resolve().parents[1] / "square_ft" / "generate_square_ft.py"
)
sq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sq)

F_TARGET = 10.0  # N
F_CONTACT = 2.0  # N, landing
GAIN = 0.0003  # m per N of force error, per step (measured stiffness ~2 N/mm)
SPEED = 0.008  # m per step along the path
APPROACH, DESCEND, WIPE, LIFT = range(4)
N_PHASES = 4


class Expert:
    def __init__(self, env, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.ctrl = env.robots[0].controller
        sim = env.sim
        markers = env.model.mujoco_arena.markers
        self.path = np.array([sim.data.body_xpos[sim.model.body_name2id(m.root_body)][:2] for m in markers])
        self.table_z = float(sim.data.body_xpos[sim.model.body_name2id(markers[0].root_body)][2])  # drawn top
        # yaw: the tool's long side across the path's overall direction
        corners = [np.array(sim.data.geom_xpos[sim.model.geom_name2id(g)]) for g in env.robots[0].gripper.important_geoms["corners"]]
        long_axis = corners[0][:2] - corners[1][:2]
        heading = self.path[-1] - self.path[0]
        across = np.array([-heading[1], heading[0]])
        dyaw = np.arctan2(across[1], across[0]) - np.arctan2(long_axis[1], long_axis[0])
        dyaw = (dyaw + np.pi / 2) % np.pi - np.pi / 2  # the tool is symmetric under 180°
        self.rot = sq.rot_z(dyaw) @ np.array(self.ctrl.ee_ori_mat)
        # dense path: 4 mm steps through the markers, 2 cm past both ends
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))]
        s = np.arange(-0.02, d[-1] + 0.02, 0.004)
        u0 = (self.path[1] - self.path[0]) / max(d[1], 1e-9)
        u1 = (self.path[-1] - self.path[-2]) / max(d[-1] - d[-2], 1e-9)
        self.dense = np.array([
            self.path[0] + u0 * si if si < 0 else self.path[-1] + u1 * (si - d[-1]) if si > d[-1]
            else np.array([np.interp(si, d, self.path[:, 0]), np.interp(si, d, self.path[:, 1])])
            for si in s
        ])
        self.progress, self.z_tgt = 0.0, None
        self.phase, self.t_phase, self.failed = APPROACH, 0, False

    def _go(self, phase):
        self.phase, self.t_phase = phase, 0

    def _force(self, block):
        w = block[-10:].mean(0)
        return float((np.array(self.ctrl.ee_ori_mat) @ w[:3])[2])  # upward force on the tool

    def act(self, block: np.ndarray) -> np.ndarray:
        p_ee, r_ee = np.array(self.ctrl.ee_pos), np.array(self.ctrl.ee_ori_mat)
        noise = 0.01
        self.t_phase += 1
        if self.phase == APPROACH:
            tgt = np.r_[self.dense[0], self.table_z + 0.06]
            a = sq.Expert._track(p_ee, r_ee, tgt, self.rot, 0.03)
            if np.linalg.norm(tgt - p_ee) < 0.01 and np.linalg.norm(sq.axisangle(self.rot @ r_ee.T)) < 0.05:
                self._go(DESCEND)
            self.failed = self.t_phase > 80
        elif self.phase == DESCEND:
            if self._force(block) > F_CONTACT:
                self.z_tgt = p_ee[2] - 0.002
                self._go(WIPE)
            step = 0.01 if p_ee[2] > self.table_z + 0.045 else 0.003  # slow where the hidden surface may be
            tgt = np.r_[self.dense[0], p_ee[2] - step]
            a = sq.Expert._track(p_ee, r_ee, tgt, self.rot, 0.01)
            self.failed = self.t_phase > 80
        elif self.phase == WIPE:
            noise = 0.005
            f = self._force(block)
            self.z_tgt -= float(np.clip(GAIN * (F_TARGET - f), -0.002, 0.002))  # too light: press deeper
            self.progress = min(self.progress + SPEED / 0.004, len(self.dense) - 1)
            xy = self.dense[int(self.progress)]
            a = sq.Expert._track(p_ee, r_ee, np.r_[xy, self.z_tgt], self.rot, 0.012)
            if self.progress >= len(self.dense) - 1 and np.linalg.norm(p_ee[:2] - xy) < 0.004:
                if len(self.env.wiped_markers) < self.env.num_markers:  # missed some: sweep back once
                    self.dense, self.progress = self.dense[::-1].copy(), 0.0
                    self.failed = self.t_phase > 150
                else:
                    self._go(LIFT)
        else:
            a = sq.Expert._track(p_ee, r_ee, p_ee + [0, 0, 0.03], self.rot, 0.02)
        a = a + noise * self.rng.standard_normal(6)
        return np.concatenate([np.clip(a, -1, 1), [-1.0]])  # 7-D like LIBERO; the wiping tool has no fingers


def state8(obs: dict) -> np.ndarray:
    return np.concatenate([obs["robot0_eef_pos"], sq.quat2axisangle(obs["robot0_eef_quat"]), np.zeros(2)])


def run_episode(env, rng, max_steps=300, keep_images=False):
    """Expert rollout from reset. Frame t: obs(t), ft observed at t (block of the previous step), action(t)."""
    from lerobot.envs.libero_ft import FTWrenchLogger
    from lerobot.envs.square_ft import SETTLE_STEPS

    env.reset()
    logger = FTWrenchLogger(env)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(np.zeros(6))
    expert = Expert(env, rng)
    frames, success, forces = [], False, []
    for _ in range(max_steps):
        block = logger.block()
        action = expert.act(block)
        fr = {"state": state8(obs), "action": action.astype(np.float32), "ft": block, "phase": expert.phase,
              "obj": np.r_[expert.path[0], expert.path[-1]].astype(np.float32)}
        if keep_images:
            fr["image"] = obs["agentview_image"][::-1, ::-1].copy()  # lerobot/libero convention (180 deg)
            fr["image2"] = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        frames.append(fr)
        obs, _, _, _ = env.step(action[:6])
        if expert.phase == WIPE:
            forces.append(env.tool_force)
        if env._check_success():
            success = True
            break
        if env.overforce or expert.failed:
            break
    logger.detach()
    info = dict(success=success, steps=len(frames), offset=env.surface.copy(), overforce=env.overforce,
                wiped=len(env.wiped_markers), forces=np.array(forces), last_phase=expert.phase,
                phases=np.bincount([fr["phase"] for fr in frames], minlength=N_PHASES))
    return frames, info


def _pilot_worker(args):
    seed, n = args
    from lerobot.envs.wipe_ft import make_wipe_ft_task

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_wipe_ft_task(None)
    return [run_episode(env, rng) for _ in range(n)]


def cmd_pilot(a):
    per = [a.episodes // a.workers + (i < a.episodes % a.workers) for i in range(a.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = [r for rs in pool.map(_pilot_worker, [(a.seed + i, n) for i, n in enumerate(per) if n]) for r in rs]
    ok = [(f, i) for f, i in res if i["success"]]
    forces = np.concatenate([i["forces"] for _, i in res if len(i["forces"])])
    print(f"{len(res)} episodes in {time.time() - t0:.0f} s | success {len(ok) / len(res):.3f} | steps median "
          f"{np.median([i['steps'] for _, i in ok]):.0f} max {max(i['steps'] for _, i in ok)} | overforce "
          f"{np.mean([i['overforce'] for _, i in res]):.3f} | wipe-phase force p10/p50/p90 "
          f"{np.round(np.percentile(forces, [10, 50, 90]), 1).tolist()} N")
    for i in [i for _, i in res if not i["success"]][:5]:
        print(f"  fail: steps {i['steps']} wiped {i['wiped']} overforce {i['overforce']} surface "
              f"{np.round([i['offset'][0] * 1000, *np.degrees(i['offset'][1:])], 1)} | phases {i['phases'].tolist()}")
    cat = lambda k: np.concatenate([np.stack([fr[k] for fr in f]) for f, _ in ok])  # noqa: E731
    np.savez(a.out, ft=np.concatenate([np.stack([fr["ft"] for fr in f][1:] + [f[-1]["ft"]]) for f, _ in ok]),
             state=cat("state"), action=cat("action"), obj=cat("obj"), phase=cat("phase"),
             episode=np.concatenate([np.full(len(f), k) for k, (f, _) in enumerate(ok)]))


def _shard_worker(args):
    seed, n_success, out = args
    from lerobot.datasets import LeRobotDataset
    from lerobot.envs.wipe_ft import TASK, make_wipe_ft_task

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_wipe_ft_task(sq.IMAGE_SIZE)
    features = {k: v for k, v in sq.FEATURES.items() if k != "expert.peg_offset"}
    features["expert.surface"] = {"dtype": "float32", "shape": (3,), "names": None}  # hidden (dz, roll, pitch)
    ds = LeRobotDataset.create(repo_id=f"local/w{seed}", fps=20, features=features, root=Path(out) / "shards" / f"w{seed}",
                               robot_type="panda", use_videos=True, image_writer_threads=4)
    n_ok = n_try = 0
    while n_ok < n_success and n_try < 3 * n_success:
        frames, info = run_episode(env, rng, keep_images=True)
        n_try += 1
        if not info["success"]:
            continue
        for fr in frames:
            ds.add_frame({
                "observation.images.image": fr["image"],
                "observation.images.image2": fr["image2"],
                "observation.state": fr["state"].astype(np.float32),
                "observation.ft_wrench": fr["ft"].astype(np.float32),
                "action": fr["action"],
                "expert.phase": np.array([fr["phase"]], dtype=np.int64),
                "expert.surface": info["offset"].astype(np.float32),
                "task": TASK,
            })
        ds.save_episode()
        n_ok += 1
    ds.finalize()
    env.close()
    return n_ok, n_try


def cmd_shards(a):
    per = [a.episodes // a.workers + (i < a.episodes % a.workers) for i in range(a.workers)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        res = list(pool.map(_shard_worker, [(a.seed + i, n, a.out) for i, n in enumerate(per) if n]))
    ok, tried = map(sum, zip(*res))
    print(f"{ok} episodes kept of {tried} in {time.time() - t0:.0f} s (expert success {ok / tried:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("pilot", "shards", "aggregate"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--repo-id", default="himchan00/wipe_ft")
    a = p.parse_args()
    {"pilot": cmd_pilot, "shards": cmd_shards, "aggregate": sq.cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
