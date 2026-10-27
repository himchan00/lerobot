"""Per-dim action scale s = std of the clean one-step drift target d over the dataset (d/s keeps its mean).

Body mode uses one value per position/rotation 3-vector + gripper so s commutes with the body rotation; impedance
targets one per Gaussian geometry.IMPEDANCE_GROUPS group over the entries the training loss sees, 1 for the logits.
Cached under $HF_LEROBOT_HOME/latent_sde_action_scale/<repo_id>/ and reused when present.
"""

import glob
import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.configs.types import FeatureType
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE

from .configuration_latent_sde import OBS_FT_WRENCH
from .geometry import (
    IMPEDANCE_GROUPS,
    action_to_endpoint,
    hand_to_grip_site,
    impedance_one_step,
    local,
    pose_from_state,
)


def resolve_action_scale(config, ds_meta) -> list[float]:
    stats = {
        key: {name: np.asarray(value).tolist() for name, value in ds_meta.stats[key].items()
              if name in ("min", "max", "mean", "std")}
        for key in (OBS_STATE, ACTION)
    }
    cache_key = {
        "repo_id": ds_meta.repo_id,
        "snapshot": ds_meta.root.name,
        "frames": ds_meta.total_frames,
        "sde_geometry": config.sde_geometry,
        "normalization": {str(k): str(getattr(v, "value", v)) for k, v in config.normalization_mapping.items()},
        "stats": stats,
        **({"impedance_targets": "one_step"} if config.impedance_targets else {}),
    }
    digest = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:16]
    path = HF_LEROBOT_HOME / "latent_sde_action_scale" / ds_meta.repo_id / f"{digest}.json"
    if path.is_file():
        scale = json.loads(path.read_text())["action_scale"]
        logging.info(f"latent_sde action_scale loaded from {path}: {scale}")
        return scale

    scale, n_frames = _sweep(config, ds_meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"action_scale": scale, "frames": n_frames, "key": cache_key}, indent=1))
    os.replace(tmp, path)  # atomic across ranks
    logging.info(f"latent_sde action_scale swept over {n_frames} frames, saved to {path}: {scale}")
    return scale


def _episodes(frames) -> list[torch.Tensor]:
    """Each episode's row indices in frame order."""
    episode = frames["episode_index"].to_numpy()
    order = np.lexsort((frames["frame_index"].to_numpy(), episode))
    return [torch.from_numpy(ep) for ep in np.split(order, np.flatnonzero(np.diff(episode[order])) + 1)]


def _sweep(config, ds_meta) -> tuple[list[float], int]:
    files = sorted(glob.glob(str(ds_meta.root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {ds_meta.root / 'data'} to sweep the action scale.")
    columns = [OBS_STATE, ACTION] + (["episode_index", "frame_index"] if config.impedance_targets else [])
    frames, ft_blocks = [], []
    for f in files:
        frames.append(pd.read_parquet(f, columns=columns))
        if config.impedance_targets:  # each frame's F/T samples; pyarrow, as pandas cannot index this column
            column = pq.read_table(f, columns=[OBS_FT_WRENCH]).column(0).combine_chunks()
            column = column.storage if isinstance(column, pa.ExtensionArray) else column
            ft_blocks.append(column.flatten().flatten().to_numpy().reshape(len(column), -1, 6))
    frames = pd.concat(frames)
    state = torch.from_numpy(np.stack(frames[OBS_STATE].to_numpy())).float()
    action = torch.from_numpy(np.stack(frames[ACTION].to_numpy())).float()

    # Same normalization as the training preprocessor.
    normalizer = NormalizerProcessorStep(
        features={OBS_STATE: config.input_features[OBS_STATE], ACTION: config.output_features[ACTION]},
        norm_map=config.normalization_mapping,
        stats=ds_meta.stats,
    )
    x = normalizer._apply_transform(state, OBS_STATE, FeatureType.STATE).double()
    a = normalizer._apply_transform(action, ACTION, FeatureType.ACTION).double()
    action_dim = a.shape[-1]
    mask = None
    if config.impedance_targets:
        # Each episode as one window, copy-padded by a frame like the dataloader (the policy works in G).
        x, ft = hand_to_grip_site(x), torch.from_numpy(np.concatenate(ft_blocks))
        targets, masks = [], []
        for ep in _episodes(frames):
            padded = [torch.cat((v[ep], v[ep][-1:])).double()[None] for v in (x, ft)]
            real = (torch.arange(len(ep) + 1) < len(ep))[None]
            target, known = impedance_one_step(padded[0], real, padded[1])
            targets.append(target[0])
            masks.append(known[0][:, None].expand_as(target[0]))
        d, mask = torch.cat(targets), torch.cat(masks)
        action_dim, groups = 14, IMPEDANCE_GROUPS  # the two logits keep 1
    elif config.sde_geometry == "so3_r3_body":
        pose = pose_from_state(x)
        d = torch.cat((local(pose, action_to_endpoint(pose, a[:, :6])), a[:, 6:7]), dim=-1)
        groups = [[0, 1, 2], [3, 4, 5], [6]]
    else:
        d = a - x
        groups = [[i] for i in range(action_dim)]

    if mask is None:
        var = d.var(dim=0, unbiased=False)
    else:  # per-dim variance over the known entries
        m = mask.to(d)
        mean = (d * m).sum(dim=0) / m.sum(dim=0)
        var = ((d - mean) ** 2 * m).sum(dim=0) / m.sum(dim=0)
    scale = [1.0] * action_dim
    for group in groups:
        s = float(var[group].mean().sqrt())
        if s > 1e-6:
            for i in group:
                scale[i] = s
    return scale, len(d)
