#!/usr/bin/env python
"""Turn native (ideal, non-reactive) demonstrations into demonstrations under perception error.

For every kept episode of a native dataset, the worker's generation chain is replayed without rendering (env A). This
reproduces the episode exactly and gives its whole scene: the model, the body placements a reset writes into the
compiled model (robosuite's door, Wipe's marker path) and the state after settling. A second environment (env B) is
set to that scene and replays the native actions up to the segment that moves relative to the contact target. There
the task's force-reactive expert takes over, with its estimate of the target off by a sampled perception error: it
touches off target and corrects from the wrench. Geometry, rendering and success rule stay the native task's; only
the expert's estimate is wrong.

usage (from this directory, latent_sde env, MUJOCO_GL=egl):
  python convert_native.py pilot --task stack_ft --per-worker 5 --workers 4 --json pilot.json
  python convert_native.py shards --task stack_ft --workers 12 --out <data>/stack_ft_conv
  python convert_native.py finish --out <data>/stack_ft_conv --repo-id himchan00/stack_ft_conv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

import generate_ft4 as G
from experts import DoorExpert, StackExpert, WipeExpert, rot_about, world_wrench
import square_expert as sq
from recorder import ShardWriter, SubstepLogger, aggregate, report, rollout, run_shards, split, state8, views

SOURCE = {"square_ft": 900000, "door_ft": 910000, "wipe_ft": 920000, "stack_ft": 940000}  # native worker seeds
CONTACT_RULES = ("stack_ft", "wipe_ft")  # tasks whose contact rules apply (env.contact_rules)
EPISODES, SPLITS = 1000, 16  # as the native datasets were generated
TRIES = 3  # perception errors tried per native episode (shards)
DATA = Path("/PublicSSD/himchan/ft4/data")  # the native datasets (pilot: check that the chain is reproduced)


# ----------------------------------------------------------------------------------------------------------------
# perception errors and the experts that act on them
# ----------------------------------------------------------------------------------------------------------------

def _square_error(rng):
    """Believed peg position minus the true one (world xy): L-inf 4-12 mm; the 41 mm peg leaves 2.25 mm per side."""
    r = rng.uniform(0.004, 0.012)
    s = rng.uniform(-r, r)
    return np.array([(r, s), (-r, s), (s, r), (s, -r)][rng.integers(4)])


def _stack_error(rng):
    """Believed green block position minus the true one: +-5 mm sideways, 4-10 mm up or down."""
    return np.r_[rng.uniform(-0.005, 0.005, 2), rng.choice([-1.0, 1.0]) * rng.uniform(0.004, 0.010)]


def _wipe_error(rng):
    """Believed table surface: height off by up to 15 mm, roll and pitch by up to 3 deg."""
    return np.r_[rng.uniform(-0.015, 0.015), rng.uniform(-np.deg2rad(3.0), np.deg2rad(3.0), 2)]


def _door_error(rng):
    """Believed hinge axis minus the true one (world xy), up to 3 cm each way."""
    return rng.uniform(-0.03, 0.03, 2)


class StackBelief(StackExpert):
    """The Stack expert seeing the green block at its true pose plus `err`."""

    err = np.zeros(3)

    def body(self, body_id):
        p, r = super().body(body_id)
        return (p + self.err, r) if body_id == self.env.cubeB_body_id else (p, r)


class DoorBelief(DoorExpert):
    """The Door expert pulling along the arc of a believed hinge. Reactive, it also yields to the door's radial push
    (admittance, accumulated up to the size of the error), so the hand comes onto the true arc. The push is taken
    relative to the first pulling steps, which carry the handle hold."""

    ADMIT, ADMIT_STEP, ADMIT_MAX, TARE_STEPS = 0.0002, 0.002, 0.03, 3  # m per N; m per step; m in all; steps
    react = True

    def act(self, block: np.ndarray) -> np.ndarray:
        if self.phase == self.ARC and self.react:
            p, r = np.array(self.ctrl.ee_pos), np.array(self.ctrl.ee_ori_mat)
            f = r @ block.mean(0)[:3]  # force of the door on the hand, world frame
            u = np.r_[(p - self.hinge)[:2], 0.0]
            u /= np.linalg.norm(u)
            f_rad = float(f @ u)
            if self.t_phase < self.TARE_STEPS:
                self.tare = f_rad if self.t_phase == 0 else self.tare + (f_rad - self.tare) / (self.t_phase + 1)
            else:
                step = np.clip(self.ADMIT * (f_rad - self.tare), -self.ADMIT_STEP, self.ADMIT_STEP) * u
                self.comp = self.comp + step
                n = np.linalg.norm(self.comp)
                self.comp *= min(1.0, self.ADMIT_MAX / max(n, 1e-9))
        return super().act(block)


def _square_expert(env, rng, err, block, react):
    e = sq.Expert(env, rng)
    e.native, e.aim = not react, e.aim + err
    e._go(sq.TRANSPORT)
    return e


def _stack_expert(env, rng, err, block, react):
    e = StackBelief(env, rng)
    e.err, e.native = err, not react
    e.go(e.TRANSPORT)
    return e


def _wipe_expert(env, rng, err, block, react):
    e = WipeExpert(env, rng)  # path from the markers, level pose from the pad
    _, r = e.ee()
    e.table_z += err[0]
    e.tilt = np.array([err[1], err[2], 0.0])  # holds the pad flush with the believed surface
    e.rot = rot_about(e.tilt) @ e.rot0
    e.tare, e.native = world_wrench(r, block), not react
    e.go(e.DESCEND)
    return e


def _door_expert(env, rng, err, block, react):
    e = DoorBelief(env, rng)
    e._set_hinge()  # the true hinge post, then the believed axis
    e.hinge = e.hinge + np.r_[err, 0.0]
    e.turn = 1.0 if np.cross([0, 0, 1.0], np.array(e.ctrl.ee_pos) - e.hinge) @ e.pull > 0 else -1.0
    e.react, e.sense = react, float(np.sign(env.sim.data.qpos[env.handle_qpos_addr]))
    e.start_pull()  # the handle is turned: the native episode hands over when it starts to pull
    return e


# task: (perception error sampler, error label size, native phase at which the expert takes over, expert factory)
CONVERT = {
    "square_ft": (_square_error, 2, sq.TRANSPORT, _square_expert),
    "stack_ft": (_stack_error, 3, StackExpert.TRANSPORT, _stack_expert),
    "wipe_ft": (_wipe_error, 3, WipeExpert.DESCEND, _wipe_expert),
    "door_ft": (_door_error, 2, DoorExpert.ARC, _door_expert),
}


# ----------------------------------------------------------------------------------------------------------------
# scenes, the native chain and the converted rollout
# ----------------------------------------------------------------------------------------------------------------

def scene(env) -> dict:
    """The model, the body placements a reset writes into the compiled model, the state after settling, and what
    lives outside the simulator state: the gripper's ramped command and the OSC's null-space posture (the joints
    after the reset's initialization noise)."""
    m, robot = env.sim.model, env.robots[0]
    return {"xml": env.model.get_xml(), "body_pos": np.array(m.body_pos), "body_quat": np.array(m.body_quat),
            "state": env.sim.get_state().flatten(), "grip": np.array(robot.gripper.current_action),
            "posture": np.array(robot.controller.initial_joint)}


def restore(env, s: dict) -> None:
    rs = np.random.get_state()  # resets draw from the global generator, which env A's chain depends on
    env.reset_from_xml_string(s["xml"])
    np.random.set_state(rs)
    env.sim.model.body_pos[:] = s["body_pos"]
    env.sim.model.body_quat[:] = s["body_quat"]
    env.sim.set_state_from_flattened(s["state"])
    env.sim.forward()
    env.robots[0].gripper.current_action = s["grip"].copy()
    env.robots[0].controller.update_initial_joints(s["posture"])


def native_chain(task: str, seed: int, n_success: int):
    """Replay a native worker's generation chain (generate_ft4._episodes) without rendering. Yields (k, scene,
    [(action, phase)]) for its k-th kept episode."""
    t, expert, make_env = G._parts(task + "_native")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(None, camera_depths=False)
    n_ok = n_try = 0
    while n_ok < n_success and n_try < t.tries * n_success:
        snap = {}

        def make_expert(e, g, snap=snap):
            snap.update(scene(e))
            return expert(e, g)

        frames, info = rollout(env, make_expert, rng, settle_action=t.settle, max_steps=t.max_steps,
                               act_dims=t.act_dims, fingers=t.fingers, record=False)
        n_try += 1
        if info["success"]:
            yield n_ok, snap, [(fr["action"], fr["phase"]) for fr in frames]
            n_ok += 1
    env.close()


class Metrics:
    """Contact quality seen during a rollout (the native rules do not check it)."""

    def __init__(self, env, task):
        self.task, self.f_max, self.gap, self.grasped, self.forces = task, 0.0, 0.0, None, []

    def step(self, env):
        if self.task == "stack_ft":
            self.f_max = max(self.f_max, env.place_force)
            g = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)
            if self.grasped and not g and env.place_force == 0.0:  # let go without support: a drop
                self.gap = max(self.gap, float(env.sim.data.body_xpos[env.cubeA_body_id][2] - 0.02 - env.true_top()))
            self.grasped = g
        elif self.task == "wipe_ft":
            self.f_max = max(self.f_max, env.tool_force)
            if env.tool_force > 0.5:
                self.forces.append(env.tool_force)

    def out(self) -> dict:
        f = np.array(self.forces)
        return {"f_max": float(self.f_max), "drop": float(self.gap),
                "in_band": float(np.mean((f >= 5) & (f <= 25))) if len(f) else float("nan")}


def convert(env, s, native, task, rng, err, react=True, record=False, takeover=True):
    """Replay `native` [(action, phase)] in env (set to scene s) until the take-over phase, then run the task's
    expert with perception error `err`; `takeover=False` replays the whole native episode. The task's contact rules
    apply. Frame t: obs(t), wrench block seen at t, action(t), as recorder.rollout."""
    from robosuite.utils.camera_utils import get_real_depth_map

    from lerobot.envs.libero_ft import FTWrenchLogger

    t = G.TASKS[task + "_native"]
    _, _, phase_h, make = CONVERT[task]
    h = next(i for i, (_, ph) in enumerate(native) if ph == phase_h) if takeover else len(native) - 1
    restore(env, s)
    env.contact_rules = task in CONTACT_RULES
    ft = FTWrenchLogger(env)
    log = SubstepLogger(env) if record else None
    obs, _, _, _ = env.step(t.settle)  # one settle step, so the first wrench block is that of a real step
    met, expert, frames, states, success = Metrics(env, task), None, [], [], False
    for k in range(t.max_steps):
        block = ft.block()
        if k <= h:
            action, phase = native[k]
        elif not takeover:
            break
        else:
            if expert is None:
                expert = make(env, rng, err, block, react)
            action, phase = expert.act(block), expert.phase
        fr = {"observation.state": state8(obs, t.fingers), "observation.ft_wrench": block, "action": action,
              "phase": phase}
        if record:
            for key, cam in (("image", "agentview"), ("image2", "robot0_eye_in_hand")):
                fr[f"observation.images.{key}"] = obs[f"{cam}_image"][::-1, ::-1].copy()
                fr[f"observation.images.{key}_depth"] = (
                    get_real_depth_map(env.sim, obs[f"{cam}_depth"])[::-1, ::-1].astype(np.float16))
            fr.update(log.frame_start())
            states.append(env.sim.get_state().flatten())
        obs, _, _, _ = env.step(action[: t.act_dims])
        if record:
            fr.update(log.frame_end())
        frames.append(fr)
        met.step(env)
        if env._check_success():
            success = True
            break
        if getattr(env, "failed", False) or expert is not None and expert.failed:
            break
    if record:
        states.append(env.sim.get_state().flatten())
        log.detach()
    ft.detach()
    info = {"success": success, "steps": len(frames), "handover": h + 1,
            "phases": np.bincount([fr["phase"] for fr in frames], minlength=16).tolist(), **met.out()}
    if record:
        info.update(xml=s["xml"], states=np.stack(states))
    return frames, info


def _rng(seed: int, k: int) -> np.random.Generator:
    return np.random.default_rng([seed, k, 7])


# ----------------------------------------------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------------------------------------------

def _native_actions(task: str, seed: int) -> dict:
    """{k: actions} of worker `seed` in the native dataset (to check that the chain is reproduced)."""
    import pyarrow.parquet as pq

    root = DATA / f"{task}_native" / "final" / "data"
    out = {}
    for f in sorted(root.rglob("*.parquet")):
        tb = pq.read_table(f, columns=["meta.source", "action"])
        src = np.array(tb.column("meta.source").to_pylist())
        act = np.array(tb.column("action").to_pylist(), dtype=np.float32)
        for k in np.unique(src[src[:, 0] == seed, 1]):
            out[int(k)] = act[(src[:, 0] == seed) & (src[:, 1] == k)]
    return out


def _pilot_worker(args):
    task, seed, n = args
    _, _, make_env = G._parts(task + "_native")
    env_b = make_env(None, camera_depths=False)
    ref = _native_actions(task, seed)
    sampler = CONVERT[task][0]
    out = []
    for k, s, native in native_chain(task, seed, n):
        a = np.array([x for x, _ in native], dtype=np.float32)
        same = k in ref and len(ref[k]) == len(a) and float(np.abs(ref[k] - a).max()) == 0.0
        rng = _rng(seed, k)
        err = sampler(rng)
        _, conv = convert(env_b, s, native, task, rng, err, react=True)
        _, blind = convert(env_b, s, native, task, _rng(seed, k), err, react=False)
        _, nat = convert(env_b, s, native, task, _rng(seed, k), err, takeover=False)
        out.append({"seed": seed, "k": k, "reproduced": same, "native_steps": len(native), "err": err.tolist(),
                    "conv": conv, "blind": blind, "native": nat})
    env_b.close()
    return out


def cmd_pilot(a):
    n = split(EPISODES, SPLITS)
    jobs = [(a.task, SOURCE[a.task] + i, min(a.per_worker, n[i])) for i in range(a.workers)]
    t0 = time.time()
    res = [r for rs in run_shards(_pilot_worker, jobs, a.workers) for r in rs]
    if a.json:
        Path(a.json).write_text(json.dumps(res))
    for kind in ("conv", "blind", "native"):
        ok = [r for r in res if r[kind]["success"]]
        print(f"{a.task} {kind:5s}: {len(ok)}/{len(res)} success | steps median "
              f"{np.median([r[kind]['steps'] for r in ok]) if ok else 0:.0f} | f_max median "
              f"{np.median([r[kind]['f_max'] for r in res]):.1f} N | drop max {max(r[kind]['drop'] for r in res) * 1000:.1f} mm"
              f" | in band {np.nanmean([r[kind]['in_band'] for r in res]):.2f}")
    print(f"reproduced {sum(r['reproduced'] for r in res)}/{len(res)} native episodes exactly; {time.time() - t0:.0f} s")


def _shard_worker(args):
    task, seed, n, out = args
    t, _, make_env = G._parts(task + "_native")
    env_b = make_env(G.IMAGE_SIZE, camera_depths=True)
    sampler, dim = CONVERT[task][:2]
    hidden = {"expert.perception_error": G.F32(dim), "expert.source": G.I64(2)}
    writer = ShardWriter(out, seed, hidden)
    n_ok = n_try = 0
    for k, s, native in native_chain(task, seed, n):
        rng = _rng(seed, k)
        for _ in range(TRIES):  # a fresh error after a failed conversion
            err = sampler(rng)
            frames, info = convert(env_b, s, native, task, rng, err, record=True)
            n_try += 1
            if info["success"]:
                break
        if info["success"]:
            writer.add(frames, info, t.instruction, {"expert.perception_error": err.astype(np.float32),
                                                     "expert.source": np.array([seed, k], dtype=np.int64)})
            np.savez_compressed(writer.side / f"ep{writer.k - 1:04d}_scene.npz", body_pos=s["body_pos"],
                                body_quat=s["body_quat"])
            n_ok += 1
    writer.close()
    env_b.close()
    return n_ok, n_try


def cmd_shards(a):
    jobs = [(a.task, SOURCE[a.task] + i, n, a.out) for i, n in enumerate(split(EPISODES, SPLITS))]
    t0 = time.time()
    report(run_shards(_shard_worker, jobs, a.workers), t0)


def cmd_finish(a):
    aggregate(a.out, a.repo_id)
    views(a.out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("pilot", "shards", "finish"))
    p.add_argument("--task", choices=sorted(CONVERT))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--per-worker", type=int, default=5, help="pilot: native episodes converted per worker")
    p.add_argument("--json")
    p.add_argument("--out")
    p.add_argument("--repo-id")
    a = p.parse_args()
    {"pilot": cmd_pilot, "shards": cmd_shards, "finish": cmd_finish}[a.cmd](a)


if __name__ == "__main__":
    main()
