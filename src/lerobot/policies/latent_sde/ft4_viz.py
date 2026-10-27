"""FT-4 eval rollouts of a latent_sde checkpoint with per-tick logs and videos, for ft4_eval_viz.ipynb.

Same protocol as trial.sh / lerobot-eval (camera_error 1 by default, contact rules, the controller the checkpoint was
trained for: impedance_targets → the Tier-3 impedance controller, else OSC), but one env at a time with the torch seed
set per episode, so the z samples differ from a batched eval of the same env seeds. Each episode is cached as
<out>/<tag>/<task>_s<seed>_cam<camera_error>.pkl and .mp4 (VP9 in MP4: VS Code's player has neither H.264 nor WebM).

Per tick t the record holds the state before the action (p, R, the measured W̄_t in world, the camera frames) and the
task state after it (task logs, e.g. Wipe's pad force); the goal is the policy's x̂_{t+1}, W̄_{t+1} (impedance) or the
OSC set point, in world.
"""

from __future__ import annotations

import base64
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML
from matplotlib.patches import Patch, Polygon
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import ft4, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import FT4Env as FT4Config
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed

FPS = 20
WIPE_CATS = ("free", "ok", "low", "high", "uneven")  # pad on the table after a tick: none, counts, < 5 N, > 25 N, CoP off
WIPE_COLORS = {"free": "0.75", "ok": "tab:green", "low": "gold", "high": "tab:red", "uneven": "tab:blue"}


# ----------------------------------------------------------------------------------------------------------------
# rollouts
# ----------------------------------------------------------------------------------------------------------------


class Runner:
    """A checkpoint with its processors, built as lerobot-eval builds them."""

    def __init__(self, ckpt: str, camera_error: float = 1.0, device: str = "cuda"):
        self.camera_error = camera_error  # scale of ft4.perturb_cameras (0: nominal cameras, as in the demos)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        cfg.pretrained_path, cfg.device = Path(ckpt), device
        self.impedance = bool(getattr(cfg, "impedance_targets", False))
        env_cfg = FT4Config(impedance=self.impedance)
        self.policy = make_policy(cfg, env_cfg=env_cfg).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=ckpt, preprocessor_overrides={
            "device_processor": {"device": device}, "rename_observations_processor": {"rename_map": {}}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg, cfg)
        self.amp = lambda: torch.autocast(device_type=torch.device(device).type) if cfg.use_amp else nullcontext()
        self.envs: dict[str, ft4.FT4Env] = {}

    def close(self):
        for env in self.envs.values():
            env.close()
        del self.policy
        torch.cuda.empty_cache()

    @torch.no_grad()
    def episode(self, task: str, seed: int) -> tuple[dict, list[np.ndarray]]:
        if task not in self.envs:
            self.envs[task] = ft4.FT4Env(task, contact_rules=True, camera_error=self.camera_error,
                                         impedance=self.impedance)
        env = self.envs[task]
        set_seed(seed)
        for p in (self.policy, self.pre, self.post):
            p.reset()
        obs, _ = env.reset(seed=seed)
        e = env._env
        log = TASK_LOGS[task](e)
        rec = {k: [] for k in ("refresh", "action", "p", "R", "w_meas", "p_goal", "R_goal", "w_goal")}
        frames, z, end, t0 = [], None, "timeout", time.time()
        for _ in range(env._max_episode_steps):
            frames.append(_frame(obs))
            p, r = np.array(obs["robot_state"]["eef"]["pos"]), np.array(obs["robot_state"]["eef"]["mat"])
            rec["p"].append(p), rec["R"].append(r), rec["w_meas"].append(env._ft_logger.world_mean())
            o = preprocess_observation(_batch(obs))
            o["task"] = [env.task_description]
            o = self.pre(self.env_pre(o))
            with torch.inference_mode(), self.amp():
                a = self.policy.select_action(o)
            rec["refresh"].append(self.policy._cached_z is not z)
            z = self.policy._cached_z
            a = self.env_post({ACTION: self.post(a)})[ACTION].to("cpu").numpy()[0]
            rec["action"].append(a)
            obs, success, failed = _step(env, a)
            if self.impedance:
                p1, r1 = p + r @ a[0:3], r @ Rotation.from_rotvec(a[3:6]).as_matrix()
                w1 = np.r_[r @ a[12:15], r @ a[15:18]]
            else:  # OSC set point
                ctrl = e.robots[0].controller
                p1, r1, w1 = np.array(ctrl.goal_pos), np.array(ctrl.goal_ori), np.full(6, np.nan)
            rec["p_goal"].append(p1), rec["R_goal"].append(r1), rec["w_goal"].append(w1)
            log.step(e)
            if success or failed:
                end = "success" if success else _cause(e)
                break
        frames.append(_frame(obs))
        rec["p"].append(np.array(obs["robot_state"]["eef"]["pos"])), rec["R"].append(np.array(obs["robot_state"]["eef"]["mat"]))
        rec["w_meas"].append(env._ft_logger.world_mean())
        rec = {k: np.array(v) for k, v in rec.items()}
        rec.update(task=task, seed=seed, impedance=self.impedance, camera_error=self.camera_error, end=end,
                   ticks=len(rec["action"]),
                   secs=time.time() - t0, **{log.key: log.result()})
        return rec, frames


def _batch(obs):
    """A single env's observation as a batch of one (as the vector env returns it, copied)."""
    return {k: _batch(v) for k, v in obs.items()} if isinstance(obs, dict) else np.array(obs)[None]


def _step(env: ft4.FT4Env, action: np.ndarray):
    """FT4Env.step without its reset at the episode's end, so the final state can still be read."""
    e = env._env
    if env.impedance:
        e.robots[0].controller.wbar = env._ft_logger.world_mean()
    raw, _, _, _ = e.step(action[: e.action_dim])
    return env._format_raw_obs(raw), bool(e._check_success()), bool(e.failed)


def _cause(e) -> str:
    flags = {"overforce": getattr(e, "overforce", False), "dropped": getattr(e, "dropped", False)}
    if isinstance(e, ft4.WipeFT4Task):
        flags.update(arm_table=e.check_contact(e.robots[0].robot_model), q_limits=e.robots[0].check_q_limits())
    return "+".join(k for k, v in flags.items() if v) or "failed"


def _frame(obs) -> np.ndarray:
    return np.concatenate([obs["pixels"]["image"][::-1, ::-1], obs["pixels"]["image2"][::-1, ::-1]], axis=1)


class _Log:
    key = ""

    def __init__(self, e):
        self.rows = []

    def step(self, e):
        self.rows.append(self.read(e))

    def result(self) -> dict:
        return {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}


class WipeLog(_Log):
    """The pad on the table after each tick: Wipe.reward's force and centre of pressure, and robosuite's marker test
    (inside the pad's corners and less than 2 cm over its bottom plane) without the contact rules."""

    key = "wipe"

    def __init__(self, e):
        super().__init__(e)
        markers = e.model.mujoco_arena.markers
        self.index = {id(m): i for i, m in enumerate(markers)}
        self.xyz = np.array([e.sim.data.body_xpos[e.sim.model.body_name2id(m.root_body)] for m in markers])
        self.wipe_tick = np.full(len(markers), -1)
        self.table_z = float(e.table_offset[2])

    def read(self, e):
        for m in e.wiped_markers:
            i = self.index[id(m)]
            if self.wipe_tick[i] < 0:
                self.wipe_tick[i] = len(self.rows)
        c1, c2, c3, c4 = (np.array(e.sim.data.geom_xpos[g]) for g in e.corner_geoms)  # (+x+y, -x+y, +x-y, -x-y)
        v1, v2 = c1 - c2, c4 - c2
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        n = np.cross(v1, v2) / (l1 * l2)
        d = self.xyz - c2
        h, u, w = d @ n, d @ v1 / l1, d @ v2 / l2
        face, axes = e.tool_frame()
        return {"force": float(e.tool_force), "cop": np.array(e.cop), "n_wiped": len(e.wiped_markers),
                "cover": (h > 0) & (h < 0.02) & (u > 0) & (u < l1) & (w > 0) & (w < l2),
                "face": face, "axes": axes, "corners": np.array([c1, c2, c4, c3])[:, :2]}

    def result(self):
        return {**super().result(), "markers": self.xyz, "wipe_tick": self.wipe_tick, "table_z": self.table_z}


class StackLog(_Log):
    key = "stack"

    def read(self, e):  # cube A's bottom over B's top (m), the force A presses B with (N)
        gap = e.sim.data.body_xpos[e.cubeA_body_id][2] - 0.02 - e.true_top()
        return {"gap": float(gap), "place_force": float(e.place_force), "grasped": bool(e.was_grasped)}


class SquareLog(_Log):
    key = "square"

    def read(self, e):  # nut − peg (m), nut height over the table (m), nut yaw (deg)
        nut = e.nuts[e.nut_id]
        b = e.obj_body_id[nut.name]
        rel = e.sim.data.body_xpos[b] - e.sim.data.body_xpos[e.peg1_body_id]
        yaw = np.degrees(Rotation.from_matrix(e.sim.data.body_xmat[b].reshape(3, 3)).as_euler("xyz")[2])
        return {"rel": np.array(rel), "height": float(e.sim.data.body_xpos[b][2] - e.table_offset[2]), "yaw": float(yaw),
                "grasped": bool(e._check_grasp(gripper=e.robots[0].gripper, object_geoms=nut.contact_geoms))}


class DoorLog(_Log):
    key = "door"

    def read(self, e):  # hinge and handle angles (rad), handle − EE distance (m)
        q = e.sim.data.qpos
        return {"hinge": float(q[e.hinge_qpos_addr]), "handle": float(q[e.handle_qpos_addr]),
                "dist": float(np.linalg.norm(e._handle_xpos - e._eef_xpos))}


TASK_LOGS = {"square_ft": SquareLog, "stack_ft": StackLog, "wipe_ft": WipeLog, "door_ft": DoorLog}


def run(ckpts: dict[str, str], tasks, seeds, out, camera_error: float = 1.0, device: str = "cuda") -> dict:
    """{tag: {task: [record per seed]}}; cached episodes of the same checkpoint and camera error are loaded instead of
    rerun."""
    recs = {}
    for tag, ckpt in ckpts.items():
        runner = None
        for task in tasks:
            for seed in seeds:
                path = Path(out) / tag / f"{task}_s{seed}_cam{camera_error:g}"
                rec = pickle.load(open(path.with_suffix(".pkl"), "rb")) if path.with_suffix(".pkl").exists() else None
                if rec is None or rec["ckpt"] != str(ckpt):
                    runner = runner or Runner(ckpt, camera_error, device)
                    rec, frames = runner.episode(task, seed)
                    rec.update(tag=tag, ckpt=str(ckpt), video=str(path.with_suffix(".mp4")))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_video(rec["video"], frames, captions(rec))
                    pickle.dump(rec, open(path.with_suffix(".pkl"), "wb"))
                    print(f"{tag} {task} s{seed} cam{camera_error:g}: {rec['end']}, {rec['ticks']} ticks "
                          f"({rec['secs']:.0f} s)")
                recs.setdefault(tag, {}).setdefault(task, []).append(rec)
        if runner is not None:
            runner.close()
    return recs


# ----------------------------------------------------------------------------------------------------------------
# videos
# ----------------------------------------------------------------------------------------------------------------


def tilt(R: np.ndarray) -> np.ndarray:
    """Angle of the EE z axis from the vertical (deg)."""
    return np.degrees(np.arccos(np.clip(np.abs(R[..., 2, 2]), 0.0, 1.0)))


def captions(r: dict) -> list[str]:
    """Frame k shows the state at tick k (before its action); its line has the task state after tick k − 1."""
    out = []
    for k in range(r["ticks"] + 1):
        s = f"{r['tag']} {r['task']} s{r['seed']}  t={k:3d}{' z' if k < r['ticks'] and r['refresh'][k] else '  '}  "
        j = k - 1
        if j >= 0 and r["task"] == "wipe_ft":
            w = r["wipe"]
            s += (f"F {w['force'][j]:5.1f} N  CoP {w['cop'][j][0] / ft4.PAD_HALF[0]:+.2f} {w['cop'][j][1] / ft4.PAD_HALF[1]:+.2f}"
                  f"  tilt {tilt(r['R'][k]):4.1f}°  wiped {w['n_wiped'][j]:3d}")
        elif j >= 0 and r["task"] == "stack_ft":
            d = r["stack"]
            s += f"gap {d['gap'][j] * 1e3:6.1f} mm  F {d['place_force'][j]:5.1f} N  grasp {int(d['grasped'][j])}"
        elif j >= 0 and r["task"] == "square_ft":
            d = r["square"]
            s += (f"nut-peg xy {np.linalg.norm(d['rel'][j][:2]) * 1e3:5.1f} mm  h {d['height'][j] * 1e3:5.1f} mm"
                  f"  grasp {int(d['grasped'][j])}")
        elif j >= 0 and r["task"] == "door_ft":
            d = r["door"]
            s += f"handle {np.degrees(d['handle'][j]):5.1f}°  hinge {np.degrees(d['hinge'][j]):5.1f}°"
        if k == r["ticks"]:
            s += f"  -> {r['end']}"
        out.append(s)
    return out


def _write_video(path: str, frames: list[np.ndarray], lines: list[str]):
    font = ImageFont.load_default(size=14)
    out = []
    for f, line in zip(frames, lines):
        im = Image.new("RGB", (f.shape[1], f.shape[0] + 32))
        im.paste(Image.fromarray(f), (0, 32))
        ImageDraw.Draw(im).text((6, 8), line, fill="white", font=font)
        out.append(np.asarray(im))
    imageio.mimwrite(path, out, fps=FPS, codec="libvpx-vp9", quality=None,
                     ffmpeg_params=["-deadline", "realtime", "-cpu-used", "8", "-crf", "34", "-b:v", "0",
                                    "-movflags", "+faststart"])


def video(r: dict, width: int = 900) -> HTML:
    data = base64.b64encode(Path(r["video"]).read_bytes()).decode()
    return HTML(f'<video controls width="{width}" src="data:video/mp4;base64,{data}"></video>')


def frames_at(r: dict, ticks, cols: int = 3):
    """The video frames (state before the tick's action) at these ticks, or at {name: tick} (e.g. wipe_events)."""
    names = {}
    for name, t in (ticks.items() if isinstance(ticks, dict) else ((None, t) for t in ticks)):
        names.setdefault(int(t), []).append(name)
    want = sorted(names)
    got = {}
    with imageio.get_reader(r["video"]) as rd:
        for i, f in enumerate(rd):
            if i in want:
                got[i] = f
            if i >= want[-1]:
                break
    rows = -(-len(want) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), squeeze=False)
    for a in axs.flat:
        a.axis("off")
    for a, (i, f) in zip(axs.flat, got.items()):
        a.imshow(f[32:]), a.set_title(", ".join([f"t={i}"] + [n for n in names[i] if n]), fontsize=10)
    fig.tight_layout()


# ----------------------------------------------------------------------------------------------------------------
# all tasks
# ----------------------------------------------------------------------------------------------------------------


def _refresh_lines(axs, r):
    for a in np.atleast_1d(axs):
        for k in np.flatnonzero(r["refresh"]):
            a.axvline(k, color="0.88", lw=0.6, zorder=0)


def _metric(r: dict) -> str:
    if r["task"] == "wipe_ft":
        return f"{r['wipe']['n_wiped'][-1]}/100 wiped"
    if r["task"] == "door_ft":
        d = r["door"]
        return f"handle max {np.degrees(d['handle'].max()):.0f}°, hinge max {np.degrees(d['hinge'].max()):.0f}°"
    if r["task"] == "stack_ft":
        return f"gap at end {r['stack']['gap'][-1] * 1e3:.0f} mm"
    d = r["square"]
    return f"nut-peg xy at end {np.linalg.norm(d['rel'][-1][:2]) * 1e3:.0f} mm, nut height {d['height'][-1] * 1e3:.0f} mm"


def summary(recs: dict) -> pd.DataFrame:
    rows = [{"ckpt": tag, "task": task, "seed": r["seed"], "cam": r["camera_error"], "end": r["end"], "ticks": r["ticks"],
             "state": _metric(r)}
            for tag, by_task in recs.items() for task, eps in by_task.items() for r in eps]
    return pd.DataFrame(rows)


def plot_episode(r: dict):
    """EE position / goal, goal error, tilt, wrench (measured W̄ vs the policy's W̄_{t+1}) and the task state.
    Goals are drawn at t + 1, where the impedance policy predicts the pose; grey lines: z resampled."""
    T, imp = r["ticks"], r["impedance"]
    t, t1 = np.arange(T + 1), np.arange(1, T + 1)
    p, R = r["p"], r["R"]
    fig, axs = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    for i, (c, col) in enumerate(zip("xyz", ("tab:blue", "tab:orange", "tab:green"))):
        axs[0].plot(t, (p[:, i] - p[0, i]) * 1e3, color=col, label=c)
        axs[0].plot(t1, (r["p_goal"][:, i] - p[0, i]) * 1e3, "--", color=col, lw=0.8)
    axs[0].set_ylabel("EE − start (mm)\n(dashed: goal)"), axs[0].legend(loc="upper right", ncol=3)
    perr = np.linalg.norm(p[1:] - r["p_goal"], axis=1) * 1e3
    rerr = np.degrees(Rotation.from_matrix(np.einsum("nji,njk->nik", R[1:], r["R_goal"])).magnitude())
    axs[1].plot(t1, perr, label="position (mm)"), axs[1].plot(t1, rerr, label="rotation (deg)")
    axs[1].set_ylabel("x(t+1) − x̂(t+1)" if imp else "x(t+1) − OSC set point"), axs[1].legend(loc="upper right")
    axs[2].plot(t, tilt(R), label="measured"), axs[2].plot(t1, tilt(r["R_goal"]), "--", label="goal")
    axs[2].set_ylabel("EE tilt from vertical (deg)"), axs[2].legend(loc="upper right")
    axs[3].plot(t, np.linalg.norm(r["w_meas"][:, :3], axis=1), label="|force| measured W̄(t)")
    if imp:
        axs[3].plot(t1, np.linalg.norm(r["w_goal"][:, :3], axis=1), "--", label="|force| policy W̄(t+1), F_d")
    axs[3].set_ylabel("N"), axs[3].legend(loc="upper right")
    TASK_PANELS[r["task"]](axs[4], r)
    _refresh_lines(axs, r)
    axs[-1].set_xlabel("tick")
    axs[0].set_title(f"{r['tag']} {r['task']} s{r['seed']}: {r['end']} at tick {T} ({_metric(r)})")
    fig.tight_layout()


def _grasp_band(a, x, on):
    a.fill_between(x, 0, 1, where=on, transform=a.get_xaxis_transform(), color="tab:green", alpha=0.12, label="grasped")


def _square_panel(a, r):
    d, x = r["square"], np.arange(1, r["ticks"] + 1)
    a.plot(x, np.linalg.norm(d["rel"][:, :2], axis=1) * 1e3, label="nut − peg xy (mm)")
    a.plot(x, d["height"] * 1e3, label="nut height (mm)")
    _grasp_band(a, x, d["grasped"]), a.legend(loc="upper right")


def _stack_panel(a, r):
    d, x = r["stack"], np.arange(1, r["ticks"] + 1)
    a.plot(x, d["gap"] * 1e3, label="A bottom − B top (mm)")
    a.plot(x, d["place_force"], label="A on B force (N)")
    _grasp_band(a, x, d["grasped"]), a.legend(loc="upper right")


def _wipe_panel(a, r):
    d, x = r["wipe"], np.arange(1, r["ticks"] + 1)
    a.axhspan(*ft4.WIPE_FORCE, color="tab:green", alpha=0.1)
    a.plot(x, d["force"], color="k", lw=0.8, label="pad force (N)")
    a.set_ylabel("N"), a.legend(loc="upper left")
    b = a.twinx()
    b.plot(x, d["n_wiped"], color="tab:purple", label="wiped"), b.set_ylabel("markers wiped")


def _door_panel(a, r):
    d, x = r["door"], np.arange(1, r["ticks"] + 1)
    a.plot(x, np.degrees(d["handle"]), label="handle (deg)"), a.plot(x, np.degrees(d["hinge"]), label="hinge (deg)")
    a.plot(x, d["dist"] * 1e3, label="handle − EE (mm)"), a.legend(loc="upper right")


TASK_PANELS = {"square_ft": _square_panel, "stack_ft": _stack_panel, "wipe_ft": _wipe_panel, "door_ft": _door_panel}


# ----------------------------------------------------------------------------------------------------------------
# Wipe
# ----------------------------------------------------------------------------------------------------------------


def wipe_categories(w: dict) -> np.ndarray:
    """Per tick (after it): free (no pad-table force), low (< 5 N), high (> 25 N), uneven (CoP outside the middle 60 %),
    else ok, i.e. covered markers count."""
    f, cop = w["force"], w["cop"]
    even = (np.abs(cop[:, 0]) < ft4.WIPE_EVEN * ft4.PAD_HALF[0]) & (np.abs(cop[:, 1]) < ft4.WIPE_EVEN * ft4.PAD_HALF[1])
    return np.select([f <= 0, f < ft4.WIPE_FORCE[0], f > ft4.WIPE_FORCE[1], ~even], ["free", "low", "high", "uneven"], "ok")


def _covered_unwiped(w: dict) -> np.ndarray:
    """(T, markers): under the pad at tick t and not wiped before t."""
    T = len(w["force"])
    wt = w["wipe_tick"][None, :]
    return w["cover"] & ((wt < 0) | (wt >= np.arange(T)[:, None]))


def wipe_funnel(eps: list[dict], plot: bool = True) -> pd.DataFrame:
    """Per episode: markers wiped; markers covered but never wiped, by the most common rule failure while covered;
    markers never covered (the path); and the share of contact ticks per category and over an unwiped marker."""
    rows = []
    for r in eps:
        w = r["wipe"]
        cat, cov, wiped = wipe_categories(w), _covered_unwiped(w), w["wipe_tick"] >= 0
        row = {"ckpt": r["tag"], "seed": r["seed"], "end": r["end"], "ticks": r["ticks"], "wiped": int(wiped.sum())}
        row.update({f"blocked_{c}": 0 for c in ("free", "low", "high", "uneven")})
        for m in np.flatnonzero(~wiped):
            ticks = np.flatnonzero(cov[:, m])
            if len(ticks):
                cs, n = np.unique(cat[ticks], return_counts=True)
                row[f"blocked_{cs[n.argmax()]}"] += 1
        row["never_covered"] = int((~wiped & ~cov.any(0)).sum())
        contact = cat != "free"
        row["contact_ticks"] = int(contact.sum())
        row.update({f"{c}_%": round(100 * float((cat == c).sum()) / max(contact.sum(), 1), 1) for c in WIPE_CATS[1:]})
        row["on_marker_%"] = round(100 * float((cov.any(1) & contact).sum()) / max(contact.sum(), 1), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    if plot and len(df):
        fig, axs = plt.subplots(1, 2, figsize=(13, 0.5 * len(df) + 1.6))
        names = [f"{a} s{b}" for a, b in zip(df["ckpt"], df["seed"])]
        cols = ["wiped", "blocked_free", "blocked_low", "blocked_high", "blocked_uneven", "never_covered"]
        colors = ["tab:green", WIPE_COLORS["free"], "gold", "tab:red", "tab:blue", "k"]
        df[cols].set_axis(names).plot.barh(stacked=True, ax=axs[0], color=colors, width=0.7)
        cols = [f"{c}_%" for c in WIPE_CATS[1:]]
        df[cols].set_axis(names).plot.barh(stacked=True, ax=axs[1], color=[WIPE_COLORS[c] for c in WIPE_CATS[1:]], width=0.7)
        for a, title in zip(axs, ("markers (of 100)", "contact ticks by category (%)")):
            a.set_title(title), a.invert_yaxis(), a.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()
    return df


def wipe_map(r: dict, ax=None, every: int = 5, legend: bool = True):
    """Top view: markers (filled by the tick they were wiped, hollow if not), the pad centre per tick coloured by its
    category, and its footprint every `every` contact ticks."""
    w = r["wipe"]
    cat = wipe_categories(w)
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    xy, wt = w["markers"][:, :2], w["wipe_tick"]
    ax.scatter(*xy[wt < 0].T, s=22, facecolors="none", edgecolors="0.3", lw=0.8, label="not wiped", zorder=3)
    sc = ax.scatter(*xy[wt >= 0].T, s=22, c=wt[wt >= 0], cmap="viridis", vmin=0, vmax=max(r["ticks"], 1), zorder=3)
    face = w["face"][:, :2]
    ax.plot(*face.T, color="0.8", lw=0.6, zorder=0)
    for c in WIPE_CATS:
        m = cat == c
        if m.any():
            ax.scatter(*face[m].T, s=5, color=WIPE_COLORS[c], label=f"pad {c} ({m.sum()})")
    for t in np.flatnonzero(cat != "free")[::every]:
        ax.add_patch(Polygon(w["corners"][t], fill=False, ec=WIPE_COLORS[cat[t]], lw=0.5, alpha=0.6))
    ax.plot(*face[0], "k^", ms=8)
    ax.set_aspect("equal"), ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)")
    ax.set_title(f"{r['tag']} s{r['seed']}: {r['end']}, {int((wt >= 0).sum())}/100, {r['ticks']} ticks", fontsize=10)
    if legend:
        ax.legend(fontsize=7, loc="best"), plt.colorbar(sc, ax=ax, label="tick wiped")


def wipe_maps(eps: list[dict], cols: int = 3):
    cols = min(cols, len(eps))
    rows = -(-len(eps) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4.6 * rows), squeeze=False)
    for a in axs.flat[len(eps):]:
        a.axis("off")
    for a, r in zip(axs.flat, eps):
        wipe_map(r, a, legend=False)
    fig.tight_layout()


def wipe_timeline(r: dict):
    """Pad force vs the 5-25 N band (and the policy's W̄_{t+1} along the pad normal), CoP vs the middle 60 %, tilt
    (measured, goal, and its components along the pad axes), pad height and markers, and the unwiped markers under
    the pad per tick coloured by the tick's category (green = counted)."""
    w, T = r["wipe"], r["ticks"]
    x, t = np.arange(1, T + 1), np.arange(T + 1)
    cat, cov = wipe_categories(w), _covered_unwiped(w)
    up = w["axes"][:, :, 2]  # pad normal (world), after each tick
    fig, axs = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    a = axs[0]
    a.axhspan(*ft4.WIPE_FORCE, color="tab:green", alpha=0.1), a.axhline(ft4.WIPE_FAIL, color="tab:red", ls=":")
    a.plot(x, w["force"], color="k", lw=0.9, label="pad−table normal force")
    a.plot(x, np.einsum("ni,ni->n", r["w_meas"][1:, :3], up), color="tab:orange", lw=0.8,
           label="measured W̄ along pad normal")
    if r["impedance"]:
        a.plot(x, np.einsum("ni,ni->n", r["w_goal"][:, :3], up), "--", color="tab:blue", lw=0.8,
               label="policy W̄(t+1) along pad normal")
    a.set_ylabel("N"), a.legend(loc="upper right", fontsize=8)
    a = axs[1]
    on = cat != "free"
    for i, (name, half) in enumerate(zip(("long", "short"), ft4.PAD_HALF)):
        a.plot(x, np.where(on, w["cop"][:, i] / half, np.nan), label=f"CoP {name} axis / half size")
    for s in (-1, 1):
        a.axhline(s * ft4.WIPE_EVEN, color="k", ls=":", lw=0.8)
    a.set_ylim(-1.1, 1.1), a.set_ylabel("CoP (pad half sizes)"), a.legend(loc="upper right", fontsize=8)
    a = axs[2]
    a.plot(t, tilt(r["R"]), label="EE tilt measured"), a.plot(x, tilt(r["R_goal"]), "--", label="EE tilt goal")
    a.plot(x, np.degrees(np.arcsin(np.clip(w["axes"][:, 2, 0], -1, 1))), lw=0.8, label="vertical · pad long axis")
    a.plot(x, np.degrees(np.arcsin(np.clip(w["axes"][:, 2, 1], -1, 1))), lw=0.8, label="vertical · pad short axis")
    a.axhline(0, color="k", lw=0.5), a.set_ylabel("deg"), a.legend(loc="upper right", fontsize=8)
    a = axs[3]
    a.plot(x, (w["face"][:, 2] - w["table_z"]) * 1e3, color="tab:brown", label="pad corner-geom centres − table top")
    a.set_ylabel("mm"), a.legend(loc="upper left", fontsize=8)
    b = a.twinx()
    b.plot(x, w["n_wiped"], color="tab:purple"), b.set_ylabel("markers wiped", color="tab:purple")
    a = axs[4]
    n = cov.sum(1)
    for c in WIPE_CATS:
        a.bar(x[cat == c], n[cat == c], width=1.0, color=WIPE_COLORS[c])
    a.legend(handles=[Patch(color=WIPE_COLORS[c], label=c) for c in WIPE_CATS], loc="upper right", fontsize=8, ncol=5)
    a.set_ylabel("unwiped markers\nunder the pad")
    _refresh_lines(axs, r)
    axs[-1].set_xlabel("tick")
    axs[0].set_title(f"{r['tag']} wipe_ft s{r['seed']}: {r['end']} at tick {T}, {_metric(r)}")
    fig.tight_layout()


def wipe_events(r: dict) -> dict:
    """Ticks worth a look: first contact, first counting contact, largest tilt and force in contact, the end."""
    w = r["wipe"]
    cat = wipe_categories(w)
    on = np.flatnonzero(cat != "free")
    ev = {}
    if len(on):
        ev["first contact"] = int(on[0]) + 1
        ok = np.flatnonzero(cat == "ok")
        if len(ok):
            ev["first ok"] = int(ok[0]) + 1
        ev["max tilt in contact"] = int(on[np.argmax(tilt(r["R"][1:])[on])]) + 1
        ev["max force"] = int(on[np.argmax(w["force"][on])]) + 1
    ev["end"] = r["ticks"]
    return ev
