"""Per-dim action scale s = std of the clean one-step drift target d over the dataset (d/s keeps its mean).

Body mode uses one value per position/rotation 3-vector + gripper so s commutes with the body rotation.
Cached under $HF_LEROBOT_HOME/latent_sde_action_scale/<repo_id>/ and reused when present.
"""

import glob
import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd
import torch

from lerobot.configs.types import FeatureType
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE

from .geometry import action_to_endpoint, local, pose_from_state


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


def _sweep(config, ds_meta) -> tuple[list[float], int]:
    files = sorted(glob.glob(str(ds_meta.root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {ds_meta.root / 'data'} to sweep the action scale.")
    frames = pd.concat([pd.read_parquet(f, columns=[OBS_STATE, ACTION]) for f in files])
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
    if config.sde_geometry == "so3_r3_body":
        pose = pose_from_state(x)
        d = torch.cat((local(pose, action_to_endpoint(pose, a[:, :6])), a[:, 6:7]), dim=-1)
        groups = [[0, 1, 2], [3, 4, 5], [6]]
    else:
        d = a - x
        groups = [[i] for i in range(action_dim)]

    var = d.var(dim=0, unbiased=False)
    scale = [1.0] * action_dim
    for group in groups:
        s = float(var[group].mean().sqrt())
        if s > 1e-6:
            for i in group:
                scale[i] = s
    return scale, len(d)
