#!/usr/bin/env python
"""Door-FT demos from a scripted F/T-reactive expert (task: lerobot/envs/door_ft.py).

The expert knows the handle pose and the door's geometry (hinge 0.255 m from the panel center), never the hinge
side. It grasps the centered bar from the front (beside its base) and pulls it straight toward the robot; from the
sideways force the door then exerts on the hand it decides which post is the hinge, and follows that hinge's arc,
switching sides if the sideways force builds up (the wrong arc). The gripper keeps its yaw while pulling: the vertical
fingers let the bar turn between them.

usage (latent_sde env):
  MUJOCO_GL=egl python generate_door_ft.py pilot --episodes 100 --workers 20 --out pilot.npz
  MUJOCO_GL=osmesa LP_NUM_THREADS=1 python generate_door_ft.py shards --episodes 1000 --workers 16 --seed 700000 --out <dir>
  python generate_door_ft.py aggregate --out <dir>
Shared pieces (dataset features, aggregation, tracking) come from ../square_ft/generate_square_ft.py.
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

HINGE_OFFSET = 0.255  # m, hinge axis from the panel center (door_ft.py)
PROBE_STEPS = 8  # straight pull before deciding the side
SIDE_BIAS = -2.5  # N, sideways force with no hinge information (measured)
ARC_STEP = 0.04  # rad per step along the hinge arc
WRONG_ARC_N = 25.0  # N of sideways force that means the arc is the wrong one
APPROACH, DESCEND, GRASP, PROBE, ARC = range(5)
PULL = PROBE  # first pulling phase
N_PHASES = 5


class Expert:
    def __init__(self, env, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.ctrl = env.robots[0].controller
        sim = env.sim
        self.handle_site = sim.model.site_name2id("Door_handle")
        r_door = sim.data.body_xmat[sim.model.body_name2id(env.door.door_body)].reshape(3, 3)
        pull = -r_door[:, 1].copy()
        pull[2] = 0
        self.pull = pull / np.linalg.norm(pull)  # away from the panel, toward the robot
        left, right = (np.array(sim.data.body_xpos[sim.model.body_name2id(f"gripper0_{n}finger")]) for n in ("left", "right"))
        v = np.asarray(self.ctrl.ee_ori_mat).T @ (left - right)
        self.finger_axis = int(np.argmax(np.abs(v[:2])))
        self.bar = r_door[:, 0]  # along the handle bar
        self.grip_offset = rng.choice([-1.0, 1.0]) * rng.uniform(0.045, 0.055)  # beside the bar's base
        self.phase, self.t_phase, self.failed, self.z_hold = APPROACH, 0, False, None
        self.f_probe, self.side, self.hinge, self.strain = [], None, None, 0

    def _rot(self, pull):
        """Gripper pointing at the door (against `pull`) with vertical fingers, the 180° choice nearer to now."""
        f = np.array([0, 0, 1.0])
        if f @ np.asarray(self.ctrl.ee_ori_mat)[:, self.finger_axis] < 0:
            f = -f
        z = -pull / np.linalg.norm(pull)
        cols = [None, None, z]
        cols[self.finger_axis] = f
        cols[1 - self.finger_axis] = np.cross(f, z) if self.finger_axis == 1 else np.cross(z, f)
        return np.stack(cols, axis=1)

    def _go(self, phase):
        self.phase, self.t_phase = phase, 0

    def _set_side(self, side):
        """Hinge axis of that post (door geometry) and the rotation sense that brings the hand toward the robot."""
        door = self.env.sim.model.body_name2id(self.env.door.door_body)
        r_door = self.env.sim.data.body_xmat[door].reshape(3, 3)
        self.side = side
        self.hinge = np.array(self.env.sim.data.body_xpos[door]) + r_door @ np.array([side * HINGE_OFFSET, 0, 0])
        rad = np.array(self.ctrl.ee_pos) - self.hinge
        self.turn = 1.0 if np.cross([0, 0, 1.0], rad) @ self.pull > 0 else -1.0

    def act(self, block: np.ndarray) -> np.ndarray:
        p, r = np.array(self.ctrl.ee_pos), np.array(self.ctrl.ee_ori_mat)
        handle = np.array(self.env.sim.data.site_xpos[self.handle_site])
        grip, noise = -1.0, 0.01
        self.t_phase += 1
        grasp = handle + self.grip_offset * self.bar
        if self.phase == APPROACH:  # in front of the bar
            r_tgt = self._rot(self.pull)
            tgt = grasp + 0.10 * self.pull
            a = sq.Expert._track(p, r, tgt, r_tgt, 0.03)
            aligned = np.linalg.norm(sq.axisangle(r_tgt @ r.T)) < (0.04 if self.t_phase < 60 else 0.1)
            if np.linalg.norm(tgt - p) < 0.008 and aligned:
                self._go(DESCEND)
            self.failed = self.t_phase > 120
        elif self.phase == DESCEND:  # forward onto the bar
            r_tgt = self._rot(self.pull)
            tgt = grasp
            a = sq.Expert._track(p, r, tgt, r_tgt, 0.01)
            if np.linalg.norm(tgt - p) < 0.004:
                self._go(GRASP)
            self.failed = self.t_phase > 60
        elif self.phase == GRASP:
            grip = 1.0
            a = sq.Expert._track(p, r, p, r, 0.01)
            if self.t_phase >= 10:
                self.z_hold, self.r_hold = p[2], r
                self._go(PROBE)
        else:
            grip, noise = 1.0, 0.005
            f = r @ block.mean(0)[:3]  # force of the door on the hand, world frame
            f_side = float(f @ np.cross([0, 0, 1.0], self.pull))  # positive: toward the left of the pull
            if self.phase == PROBE:  # straight pull; the door's sideways reaction shows the hinge side
                if self.t_phase > 2:
                    self.f_probe.append(f_side)
                if self.t_phase >= PROBE_STEPS:
                    self._set_side(1 if np.mean(self.f_probe) > SIDE_BIAS else -1)
                    self._go(ARC)
                tgt = np.r_[p[:2] + 0.01 * self.pull[:2], self.z_hold]
            else:  # rotate the hand about the chosen hinge, toward the robot
                rad = p - self.hinge
                tangent = np.cross([0, 0, self.turn], rad)
                lateral = abs(float(f @ np.cross([0, 0, 1.0], tangent / np.linalg.norm(tangent))))
                self.strain = self.strain + 1 if lateral > WRONG_ARC_N else 0
                if self.strain >= 3:  # the other post is the hinge
                    self._set_side(-self.side)
                    self.strain, self.flips = 0, getattr(self, "flips", 0) + 1
                    rad = p - self.hinge
                c, s = np.cos(self.turn * ARC_STEP), np.sin(self.turn * ARC_STEP)
                tgt = self.hinge + np.array([c * rad[0] - s * rad[1], s * rad[0] + c * rad[1], 0.0])
                tgt[2] = self.z_hold
                self.failed = self.t_phase > 150 or getattr(self, "flips", 0) > 2
            a = sq.Expert._track(p, r, tgt, self.r_hold, 0.015)
        a = a + noise * self.rng.standard_normal(6)
        return np.concatenate([np.clip(a, -1, 1), [grip]])


def run_episode(env, rng, max_steps=300, keep_images=False):
    """Expert rollout from reset. Frame t: obs(t), ft observed at t (block of the previous step), action(t)."""
    from lerobot.envs.libero_ft import FTWrenchLogger
    from lerobot.envs.square_ft import SETTLE_ACTION, SETTLE_STEPS

    env.reset()
    logger = FTWrenchLogger(env)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(SETTLE_ACTION)
    expert = Expert(env, rng)
    frames, success, forces = [], False, []
    for _ in range(max_steps):
        block = logger.block()
        action = expert.act(block)
        fr = {"state": sq.state8(obs), "action": action.astype(np.float32), "ft": block, "phase": expert.phase,
              "obj": np.r_[env.sim.data.site_xpos[expert.handle_site], env.hinge_side].astype(np.float32)}
        if keep_images:
            fr["image"] = obs["agentview_image"][::-1, ::-1].copy()  # lerobot/libero convention (180 deg)
            fr["image2"] = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        frames.append(fr)
        obs, _, _, _ = env.step(action)
        if expert.phase >= PULL:
            forces.append(env.hand_force)
        if env._check_success():
            success = True
            break
        if env.overforce or expert.failed:
            break
    logger.detach()
    info = dict(success=success, steps=len(frames), side=env.hinge_side, overforce=env.overforce,
                hinge=float(env.sim.data.qpos[env.hinge_qpos_addr]), forces=np.array(forces), last_phase=expert.phase,
                phases=np.bincount([fr["phase"] for fr in frames], minlength=N_PHASES))
    return frames, info


def _pilot_worker(args):
    seed, n = args
    from lerobot.envs.door_ft import make_door_ft_task

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_door_ft_task(None)
    return [run_episode(env, rng) for _ in range(n)]


def cmd_pilot(a):
    per = [a.episodes // a.workers + (i < a.episodes % a.workers) for i in range(a.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = [r for rs in pool.map(_pilot_worker, [(a.seed + i, n) for i, n in enumerate(per) if n]) for r in rs]
    ok = [(f, i) for f, i in res if i["success"]]
    forces = np.concatenate([i["forces"] for _, i in res if len(i["forces"])])
    print(f"{len(res)} episodes in {time.time() - t0:.0f} s | success {len(ok) / max(len(res), 1):.3f} | steps median "
          f"{np.median([i['steps'] for _, i in ok]) if ok else 0:.0f} | overforce {np.mean([i['overforce'] for _, i in res]):.3f}"
          f" | success by side: left {np.mean([i['success'] for _, i in res if i['side'] < 0]):.2f} right "
          f"{np.mean([i['success'] for _, i in res if i['side'] > 0]):.2f} | pull force p50/p90/max "
          f"{np.round(np.percentile(forces, [50, 90, 100]), 1).tolist() if len(forces) else []} N")
    for i in [i for _, i in res if not i["success"]][:6]:
        print(f"  fail: steps {i['steps']} side {i['side']} hinge {i['hinge']:.2f} overforce {i['overforce']} "
              f"last phase {i['last_phase']} phases {i['phases'].tolist()}")
    if ok:
        cat = lambda k: np.concatenate([np.stack([fr[k] for fr in f]) for f, _ in ok])  # noqa: E731
        np.savez(a.out, ft=np.concatenate([np.stack([fr["ft"] for fr in f][1:] + [f[-1]["ft"]]) for f, _ in ok]),
                 state=cat("state"), action=cat("action"), obj=cat("obj"), phase=cat("phase"),
                 episode=np.concatenate([np.full(len(f), k) for k, (f, _) in enumerate(ok)]))


def _shard_worker(args):
    seed, n_success, out = args
    from lerobot.datasets import LeRobotDataset
    from lerobot.envs.door_ft import TASK, make_door_ft_task

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_door_ft_task(sq.IMAGE_SIZE)
    features = {k: v for k, v in sq.FEATURES.items() if k != "expert.peg_offset"}
    features["expert.hinge_side"] = {"dtype": "int64", "shape": (1,), "names": None}  # hidden, for analysis only
    ds = LeRobotDataset.create(repo_id=f"local/w{seed}", fps=20, features=features, root=Path(out) / "shards" / f"w{seed}",
                               robot_type="panda", use_videos=True, image_writer_threads=4)
    n_ok = n_try = 0
    quota = {-1: n_success // 2, 1: n_success - n_success // 2}  # as many left as right hinges
    while n_ok < n_success and n_try < 8 * n_success:
        frames, info = run_episode(env, rng, keep_images=True)
        n_try += 1
        if not info["success"] or quota[info["side"]] == 0:
            continue
        quota[info["side"]] -= 1
        for fr in frames:
            ds.add_frame({
                "observation.images.image": fr["image"],
                "observation.images.image2": fr["image2"],
                "observation.state": fr["state"].astype(np.float32),
                "observation.ft_wrench": fr["ft"].astype(np.float32),
                "action": fr["action"],
                "expert.phase": np.array([fr["phase"]], dtype=np.int64),
                "expert.hinge_side": np.array([info["side"]], dtype=np.int64),
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
    p.add_argument("--repo-id", default="himchan00/door_ft")
    a = p.parse_args()
    {"pilot": cmd_pilot, "shards": cmd_shards, "aggregate": sq.cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
