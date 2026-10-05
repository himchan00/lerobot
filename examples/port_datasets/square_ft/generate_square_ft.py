#!/usr/bin/env python
"""Square-FT demos from a scripted F/T-reactive expert (task: lerobot/envs/square_ft.py).

The expert sees the true nut pose and the drawn peg, never the hidden peg offset. It aims the nut at the drawn
peg and lowers it. When the nut lands on the peg top, it presses for two steps, reads the support's center of
pressure (CoP) from the wrist wrench, then slides the nut toward the CoP while still touching the peg top,
re-reading the CoP from the current wrench at every step, until the nut drops into the hole.

usage (latent_sde env, MUJOCO_GL=egl):
  python generate_square_ft.py pilot  --episodes 100 --workers 20 --out pilot.npz        # no images; ft_info.py input
  python generate_square_ft.py shards --episodes 1000 --workers 16 --seed 500000 --out <dir>  # LeRobot shards, ~2.5 GB RAM per worker
  python generate_square_ft.py aggregate --out <dir>                                     # <dir>/final
"""

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HOLE_HALF = 0.02275  # square nut hole half-width
NUT_HALF_THICK = 0.01
WINDOW = 10  # physics samples (of 25) averaged for the wrench
CONTACT_N = 2.5  # upward force on the gripper that counts as landing on the peg top
SLIDE_LEAD = 0.006  # nut target ahead of the nut along the CoP direction
SLIDE_PUSH = 0.003  # nut target below the landing height (light preload that keeps the CoP readable)
DROP = 0.004  # the nut is this far below its landing height: it is in the hole
SUPPORT_LOST = 0.8  # upward force (N) below which the nut no longer rests on the peg top

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


def quat2axisangle(q: np.ndarray) -> np.ndarray:  # xyzw, as LiberoProcessorStep
    w = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    return np.zeros(3) if den < 1e-10 else q[:3] * 2.0 * np.arccos(w) / den


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
        self.aim = peg[:2].copy()  # where the expert believes the hole is: the drawn peg
        left, right = (
            np.array(sim.data.body_xpos[sim.model.body_name2id(f"gripper0_{n}finger")]) for n in ("left", "right")
        )
        v = np.asarray(self.ctrl.ee_ori_mat).T @ (left - right)
        self.finger_axis = int(np.argmax(np.abs(v[:2])))  # finger closing axis in the EE frame
        self.grasp_shift = rng.uniform(-0.005, 0.005)  # along the handle
        self.phase, self.t_phase = APPROACH, 0
        self.tare, self.contact_z, self.dir, self.presses, self.misses = np.zeros(6), None, None, 0, 0
        self.failed = False  # gave up (grasp lost, jammed); such episodes are dropped
        self.cops = []  # per press: (true offset from the nut, CoP from the hole center or None)

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
        if f @ r_ee[:, self.finger_axis] < 0:
            f = -f
        z = np.array([0, 0, -1.0])
        cols = [None, None, z]
        cols[self.finger_axis] = f
        cols[1 - self.finger_axis] = np.cross(f, z) if self.finger_axis == 1 else np.cross(z, f)
        return np.stack(cols, axis=1)

    def _handle_point(self) -> np.ndarray:
        _, r_nut = self._nut()
        return np.array(self.env.sim.data.site_xpos[self.handle_site]) + r_nut[:, 0] * self.grasp_shift

    def _nut_target(self, nut_xy: np.ndarray, nut_z: float):
        """EE pose that puts the nut center at (nut_xy, nut_z) with the hole square to the peg."""
        p_ee, r_ee = self._ee()
        p_nut, r_nut = self._nut()
        yaw = np.arctan2(r_nut[1, 0], r_nut[0, 0])
        dyaw = -((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)
        rz = rot_z(dyaw)
        rel = rz @ (p_ee - p_nut)
        return np.array([nut_xy[0] + rel[0], nut_xy[1] + rel[1], nut_z + rel[2]]), rz @ r_ee, abs(dyaw)

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
            if np.linalg.norm(p_tgt - p_ee) < 0.01 and (aligned or self.t_phase > 60):
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
            p_tgt, _, _ = self._nut_target(p_nut[:2], self.peg_top + NUT_HALF_THICK + 0.04)
            p_tgt[:2] = p_ee[:2]
            a = self._track(p_ee, r_ee, p_tgt, r_ee, 0.03)
            if p_nut[2] > self.peg_top + NUT_HALF_THICK + 0.025:
                self._go(TRANSPORT)
            self.failed = self.t_phase > 60  # grasp missed
        elif self.phase == TRANSPORT:
            z = self.peg_top + NUT_HALF_THICK + 0.025
            p_tgt, r_tgt, dyaw = self._nut_target(self.aim, z)
            near = np.linalg.norm(p_nut[:2] - self.aim) < 0.01
            noise = 0.01 if near else noise
            a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.01 if near else 0.03)
            if np.linalg.norm(p_nut[:2] - self.aim) < 0.002 and dyaw < 0.02 and abs(p_nut[2] - z) < 0.005:
                self._go(PROBE)
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
                true = self.env.peg_offset + np.array(self.env.sim.data.body_xpos[self.env.peg1_body_id][:2])
                self.cops.append((true - p_nut[:2], c))
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
            if self.t_phase <= 6 and p_nut[2] > self.peg_top and dw[2] > CONTACT_N:  # caught on the rim
                self.contact_z = p_nut[2]
                a = self._press(p_ee, r_ee)
            else:  # gently first, while the nut may still sit on the rim
                p_tgt, r_tgt, _ = self._nut_target(p_nut[:2], self.env.table_offset[2] + 0.012)
                a = self._track(p_ee, r_ee, p_tgt, r_tgt, 0.003 if self.t_phase <= 6 else 0.01)
                self.failed = self.t_phase > 80  # jammed
        a = a + noise * self.rng.standard_normal(6)
        return np.concatenate([np.clip(a, -1, 1), [grip]])


def state8(obs: dict) -> np.ndarray:
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])


def run_episode(env, rng, max_steps=400, keep_images=False):
    """Expert rollout from reset. Frame t: obs(t), ft observed at t (block of the previous step), action(t)."""
    from lerobot.envs.libero_ft import FTWrenchLogger
    from lerobot.envs.square_ft import SETTLE_ACTION, SETTLE_STEPS

    env.reset()
    logger = FTWrenchLogger(env)
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(SETTLE_ACTION)
    expert = Expert(env, rng)
    frames, success = [], False
    for _ in range(max_steps):
        block = logger.block()
        action = expert.act(block)
        p_nut, r_nut = expert._nut()
        fr = {
            "state": state8(obs),
            "action": action.astype(np.float32),
            "ft": block,
            "phase": expert.phase,
            "obj": np.concatenate([p_nut, r_nut[:, 0], env.sim.data.body_xpos[env.peg1_body_id]]).astype(np.float32),
        }
        if keep_images:
            fr["image"] = obs["agentview_image"][::-1, ::-1].copy()  # lerobot/libero convention (180 deg)
            fr["image2"] = obs["robot0_eye_in_hand_image"][::-1, ::-1].copy()
        frames.append(fr)
        obs, _, _, _ = env.step(action)
        if env._check_success():
            success = True
            break
        if expert.failed:
            break
    logger.detach()
    info = dict(
        success=success,
        steps=len(frames),
        offset=env.peg_offset.copy(),
        cops=expert.cops,
        phases=np.bincount([fr["phase"] for fr in frames], minlength=N_PHASES),
        last_phase=expert.phase,
    )
    return frames, info


def _pilot_worker(args):
    seed, n, kp_pos, kp_rot = args
    from lerobot.envs import square_ft

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = square_ft.make_square_ft_task(None, kp_pos, kp_rot)
    return [run_episode(env, rng) for _ in range(n)]


def cmd_pilot(a):
    per = [a.episodes // a.workers + (i < a.episodes % a.workers) for i in range(a.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        jobs = [(a.seed + i, n, a.kp_pos, a.kp_rot) for i, n in enumerate(per) if n]
        res = [r for rs in pool.map(_pilot_worker, jobs) for r in rs]
    ok = [(f, i) for f, i in res if i["success"]]
    slide = np.array([i["phases"][PRESS] + i["phases"][SLIDE] for _, i in ok])
    print(
        f"{len(res)} episodes in {time.time() - t0:.0f} s | success {len(ok) / len(res):.3f} | steps median "
        f"{np.median([i['steps'] for _, i in ok]):.0f} max {max(i['steps'] for _, i in ok)} | press+slide steps median "
        f"{np.median(slide):.0f} p90 {np.percentile(slide, 90):.0f}, episodes without contact {np.mean(slide == 0):.2f}"
    )
    cops = [(t, c) for _, i in res for t, c in i["cops"]]
    if cops:
        valid = [(t, c) for t, c in cops if c is not None]
        true, c = np.array([t for t, _ in valid]), np.array([c for _, c in valid])
        cos = (true * c).sum(1) / (np.linalg.norm(true, axis=1) * np.linalg.norm(c, axis=1) + 1e-12)
        print(
            f"  presses with a plausible CoP {len(valid) / len(cops):.2f} | CoP direction vs true offset: cos "
            f"median {np.median(cos):.3f}, p10 {np.percentile(cos, 10):.3f}, frac < 0 {np.mean(cos < 0):.2f}"
        )
    for i in [i for _, i in res if not i["success"]][:5]:
        print(
            f"  fail: steps {i['steps']} offset {np.round(i['offset'] * 1000, 2)} mm | last phase {i['last_phase']} "
            f"| steps per phase {i['phases'].tolist()}"
        )
    cat = lambda k: np.concatenate([np.stack([fr[k] for fr in f]) for f, _ in ok])  # noqa: E731
    # `ft` as in replay_ft.py: the block logged during step t (seen at t + 1); the last one is a filler
    np.savez(
        a.out,
        ft=np.concatenate([np.stack([fr["ft"] for fr in f][1:] + [f[-1]["ft"]]) for f, _ in ok]),
        state=cat("state"),
        action=cat("action"),
        obj=cat("obj"),
        phase=cat("phase"),
        episode=np.concatenate([np.full(len(f), k) for k, (f, _) in enumerate(ok)]),
        offset=np.array([i["offset"] for _, i in ok]),
    )


IMAGE_SIZE = 256
FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channels"]},
    "observation.images.image2": {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "observation.ft_wrench": {"dtype": "float32", "shape": (25, 6), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
    "expert.phase": {"dtype": "int64", "shape": (1,), "names": None},
    "expert.peg_offset": {"dtype": "float32", "shape": (2,), "names": None},  # hidden offset, for analysis only
}


def _shard_worker(args):
    seed, n_success, out = args
    from lerobot.datasets import LeRobotDataset
    from lerobot.envs.square_ft import TASK, make_square_ft_task

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_square_ft_task(IMAGE_SIZE)
    ds = LeRobotDataset.create(
        repo_id=f"local/w{seed}",
        fps=20,
        features=FEATURES,
        root=Path(out) / "shards" / f"w{seed}",
        robot_type="panda",
        use_videos=True,
        image_writer_threads=4,
    )
    n_ok = n_try = 0
    while n_ok < n_success and n_try < 3 * n_success:
        frames, info = run_episode(env, rng, keep_images=True)
        n_try += 1
        if not info["success"]:
            continue
        for fr in frames:
            ds.add_frame(
                {
                    "observation.images.image": fr["image"],
                    "observation.images.image2": fr["image2"],
                    "observation.state": fr["state"].astype(np.float32),
                    "observation.ft_wrench": fr["ft"].astype(np.float32),
                    "action": fr["action"],
                    "expert.phase": np.array([fr["phase"]], dtype=np.int64),
                    "expert.peg_offset": info["offset"].astype(np.float32),
                    "task": TASK,
                }
            )
        ds.save_episode()
        n_ok += 1
    ds.finalize()
    env.close()
    return n_ok, n_try


def cmd_shards(a):
    """One LeRobot dataset per worker under <out>/shards (successful episodes only)."""
    per = [a.episodes // a.workers + (i < a.episodes % a.workers) for i in range(a.workers)]
    t0 = time.time()
    # not mp.Pool: its daemonic workers cannot start the dataset writer's processes
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as pool:
        res = list(pool.map(_shard_worker, [(a.seed + i, n, a.out) for i, n in enumerate(per) if n]))
    ok, tried = map(sum, zip(*res))
    print(f"{ok} episodes kept of {tried} in {time.time() - t0:.0f} s (expert success {ok / tried:.3f})")


def cmd_aggregate(a):
    """<out>/final from the shards, with the Array2D workaround of examples/port_datasets/libero_hf."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "libero_hf" / "regenerate_libero_hf.py"
    spec = importlib.util.spec_from_file_location("regenerate_libero_hf", path)
    regen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(regen)
    regen.cmd_aggregate(argparse.Namespace(out_dir=a.out, repo_id=a.repo_id))



def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("pilot", "shards", "aggregate"))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kp-pos", type=float, default=150.0)
    p.add_argument("--kp-rot", type=float, default=150.0)
    p.add_argument("--out", required=True)
    p.add_argument("--repo-id", default="himchan00/square_ft")
    a = p.parse_args()
    {"pilot": cmd_pilot, "shards": cmd_shards, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()
