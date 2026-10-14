#!/usr/bin/env python
"""FT-4 native demonstrations: robosuite Square, Stack, Wipe and Door with ideal (non-reactive) experts.

Each worker seed records its successful episodes with recorder.py: two RGB-D cameras, state, the wrist wrench block
and the 500 Hz logs per frame, and a side file (model and states) per episode. Square uses the expert of
square_expert.py, Stack, Wipe and Door those of experts.py, all on lerobot.envs.ft4 scenes with `native=True`. Native
Stack and Wipe keep only episodes that also pass the contact rules. convert_native.py turns these demonstrations into
perception-error ones, build_ft4.py merges the four tasks.

usage (from this directory, latent_sde env, MUJOCO_GL=egl):
  python generate_ft4.py pilot --task stack_ft_native --episodes 100 --workers 16 --seed 7000
  python generate_ft4.py shards --task stack_ft_native --episodes 1000 --splits 16 --workers 14 --seed 940000 --out <dir>
  python generate_ft4.py finish --out <dir> --repo-id himchan00/stack_ft_native
Seeds of the datasets (16 splits): square 900000, door 910000, wipe 920000, stack 940000.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from recorder import IMAGE_SIZE, ShardWriter, aggregate, load_module, report, rollout, run_shards, split, views

F32 = lambda *s: {"dtype": "float32", "shape": s, "names": None}  # noqa: E731
I64 = lambda *s: {"dtype": "int64", "shape": s, "names": None}  # noqa: E731
GRIP_OPEN = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)


@dataclass
class Task:
    generator: str  # module with the expert class
    env: tuple  # (module, factory, *args); factory(*args, image_size, camera_depths=..., **env_kwargs) -> robosuite env
    instruction: str
    max_steps: int
    expert: str = "Expert"  # class name in `generator`
    env_kwargs: dict = field(default_factory=dict)
    settle: np.ndarray = field(default_factory=lambda: GRIP_OPEN)
    tries: int = 3  # give up after tries * episodes attempts
    hidden: dict = field(default_factory=dict)  # expert.* label features (none in FT-4)
    labels: Callable = lambda env: {}
    act_dims: int = 7
    fingers: bool = True


TASKS = {
    "square_ft_native": Task("ft4/square_expert.py", ("lerobot.envs.ft4", "make_ft4_task", "square_ft"),
                             "put the square nut on the square peg", 400, env_kwargs={"native": True}),
    "door_ft_native": Task("ft4/experts.py", ("lerobot.envs.ft4", "make_ft4_task", "door_ft"),
                           "open the door", 300, "DoorExpert", {"native": True}, tries=8),
    "stack_ft_native": Task("ft4/experts.py", ("lerobot.envs.ft4", "make_ft4_task", "stack_ft"),
                            "stack the red block on the green block", 300, "StackExpert", {"native": True}),
    "wipe_ft_native": Task("ft4/experts.py", ("lerobot.envs.ft4", "make_ft4_task", "wipe_ft"),
                           "wipe the marked path on the table", 600, "WipeExpert",
                           {"native": True, "contact_rules": True}, np.zeros(6), act_dims=6, fingers=False),
}


def _parts(name: str):
    import importlib

    t = TASKS[name]
    gen = load_module(Path(t.generator).stem, t.generator)
    make_env = partial(getattr(importlib.import_module(t.env[0]), t.env[1]), *t.env[2:], **t.env_kwargs)
    return t, getattr(gen, t.expert), make_env


def _episodes(name, seed, n_success, record, stop_after=None, rollouts=None):
    """Yield (frames or None, info, env) per rollout of one worker seed; frames is None for rollouts that are not
    kept. `rollouts` runs exactly that many rollouts instead (pilot)."""
    t, expert, make_env = _parts(name)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(IMAGE_SIZE if record else None, camera_depths=record)
    n_ok = n_try = 0
    while (n_try < rollouts) if rollouts else (n_ok < n_success and n_try < t.tries * n_success and n_ok != stop_after):
        frames, info = rollout(env, expert, rng, settle_action=t.settle, max_steps=t.max_steps, act_dims=t.act_dims,
                               fingers=t.fingers, record=record)
        n_try += 1
        if not info["success"]:
            yield None, info, env
            continue
        n_ok += 1
        yield frames, info, env
    env.close()


def _shard_worker(args):
    name, seed, n, out, stop_after = args
    t = TASKS[name]
    writer = ShardWriter(out, seed, t.hidden)
    n_ok = n_try = 0
    for frames, info, env in _episodes(name, seed, n, True, stop_after):
        n_try += 1
        if frames is not None:
            writer.add(frames, info, t.instruction, t.labels(env))
            n_ok += 1
    writer.close()
    return n_ok, n_try


def _pilot_worker(args):
    name, seed, n = args
    return [{"success": info["success"], "steps": info["steps"], "phases": info["phases"].tolist()}
            for _, info, _ in _episodes(name, seed, n, False, rollouts=n)]


def cmd_pilot(a):
    jobs = [(a.task, a.seed + i, n) for i, n in enumerate(split(a.episodes, a.workers)) if n]
    t0 = time.time()
    res = [r for rs in run_shards(_pilot_worker, jobs, a.workers) for r in rs]
    ok = [r for r in res if r["success"]]
    print(f"{a.task}: {len(res)} rollouts in {time.time() - t0:.0f} s | success {len(ok) / len(res):.3f} | steps "
          f"median {np.median([r['steps'] for r in ok]) if ok else 0:.0f}")


def _jobs(a, stop_after=None):
    return [(a.task, a.seed + i, n, a.out, stop_after) for i, n in enumerate(split(a.episodes, a.splits)) if n]


def cmd_shards(a):
    t0 = time.time()
    report(run_shards(_shard_worker, _jobs(a), a.workers), t0)


def cmd_verify(a):
    """Regenerate the first `--check` episodes of the first worker seed and compare them with an existing dataset."""
    import pandas as pd
    import pyarrow.parquet as pq

    job = _jobs(a, stop_after=a.check)[0]
    t0 = time.time()
    report(run_shards(_shard_worker, [job], 1), t0)
    read = lambda root, cols: pd.concat([pq.read_table(f, columns=cols).to_pandas()  # noqa: E731
                                         for f in sorted((Path(root) / "data").rglob("*.parquet"))])
    new = read(Path(a.out) / "shards" / f"w{a.seed}", ["episode_index", "action", "observation.state"])
    old = read(a.ref, ["episode_index", "action", "observation.state"])
    for e in range(a.check):
        n, o = new[new.episode_index == e], old[old.episode_index == e]
        da = np.abs(np.stack(n.action) - np.stack(o.action)).max() if len(n) == len(o) else float("nan")
        ds = np.abs(np.stack(n["observation.state"]) - np.stack(o["observation.state"])).max() if len(n) == len(o) else float("nan")
        print(f"episode {e}: frames new {len(n)} old {len(o)} | max |action diff| {da:.2e} | max |state diff| {ds:.2e}")


def cmd_finish(a):
    aggregate(a.out, a.repo_id)
    views(a.out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("pilot", "shards", "verify", "finish"))
    p.add_argument("--task", choices=sorted(TASKS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--splits", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--repo-id")
    p.add_argument("--ref", help="verify: an existing dataset root (its final/ view)")
    p.add_argument("--check", type=int, default=3)
    a = p.parse_args()
    {"pilot": cmd_pilot, "shards": cmd_shards, "verify": cmd_verify, "finish": cmd_finish}[a.cmd](a)


if __name__ == "__main__":
    main()
