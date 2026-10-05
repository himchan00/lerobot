"""Per-axis F/T scale for `ft_preprocess="forge"`: p90 of |w| over the training data's contact frames.

w is the FORGE EMA of `observation.ft_wrench` over the last `ft_history` control steps, exactly as in
`LatentSDEModel.ft_features`; contact means |F| > 0.5 N. tanh(w / scale) then maps a typical contact to ~0.5
and saturates impacts. Cached under $HF_LEROBOT_HOME/latent_sde_ft_scale/<repo_id>/ and reused when present.
"""

import glob
import hashlib
import json
import logging
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.utils.constants import HF_LEROBOT_HOME

from .configuration_latent_sde import OBS_FT_WRENCH

CONTACT_N = 0.5
QUANTILE = 0.9


def forge_ema_weights(ft_history: int) -> torch.Tensor:
    """FORGE smooths with 0.25 per 1/120 s physics step: the same 29 ms time constant at 2 ms substeps."""
    alpha = 1 - 0.75 ** (0.002 * 120)
    n = 25 * ft_history
    weights = alpha * (1 - alpha) ** torch.arange(n - 1, -1, -1, dtype=torch.float64)
    weights[0] = (1 - alpha) ** (n - 1)  # the EMA starts from the oldest sample
    return weights


def resolve_ft_scale(config, ds_meta) -> list[float]:
    stats = {k: np.asarray(v).tolist() for k, v in ds_meta.stats[OBS_FT_WRENCH].items() if k in ("mean", "std")}
    cache_key = {
        "repo_id": ds_meta.repo_id,
        "snapshot": ds_meta.root.name,
        "frames": ds_meta.total_frames,
        "ft_history": config.ft_history,
        "contact_n": CONTACT_N,
        "quantile": QUANTILE,
        "stats": stats,
    }
    digest = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:16]
    path = HF_LEROBOT_HOME / "latent_sde_ft_scale" / ds_meta.repo_id / f"{digest}.json"
    if path.is_file():
        scale = json.loads(path.read_text())["ft_scale"]
        logging.info(f"latent_sde ft_scale loaded from {path}: {scale}")
        return scale

    scale, n_contact = _sweep(config, ds_meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"ft_scale": scale, "contact_frames": n_contact, "key": cache_key}, indent=1))
    os.replace(tmp, path)  # atomic across ranks
    logging.info(f"latent_sde ft_scale swept over {n_contact} contact frames, saved to {path}: {scale}")
    return scale


def _sweep(config, ds_meta) -> tuple[list[float], int]:
    files = sorted(glob.glob(str(ds_meta.root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files under {ds_meta.root / 'data'} to sweep the F/T scale.")
    # pyarrow, not pandas: pandas concat breaks on the Array2D extension dtype
    shape = tuple(ds_meta.features[OBS_FT_WRENCH]["shape"])
    ft, episode = [], []
    for f in files:
        table = pq.read_table(f, columns=[OBS_FT_WRENCH, "episode_index"])
        col = table.column(OBS_FT_WRENCH).combine_chunks()
        col = col.storage if isinstance(col, pa.ExtensionArray) else col
        while pa.types.is_list(col.type) or pa.types.is_fixed_size_list(col.type):
            col = col.flatten()
        ft.append(col.to_numpy(zero_copy_only=False).reshape(-1, *shape).astype(np.float32))
        episode.append(table.column("episode_index").to_numpy())
    ft, episode = np.concatenate(ft), np.concatenate(episode)  # ft: (N, 25, 6)

    # Windows of the last ft_history frames, repeating an episode's first frame like the dataset padding.
    k = np.arange(len(ft))
    first = np.r_[0, np.nonzero(episode[1:] != episode[:-1])[0] + 1]
    start = first[np.searchsorted(first, k, side="right") - 1]
    idx = np.maximum(k[:, None] + np.arange(1 - config.ft_history, 1), start[:, None])
    weights = forge_ema_weights(config.ft_history).numpy().astype(np.float32)
    w = np.concatenate([np.einsum("nsd,s->nd", ft[i].reshape(len(i), -1, 6), weights) for i in np.array_split(idx, 64)])

    contact = np.linalg.norm(w[:, :3], axis=1) > CONTACT_N
    if not contact.any():
        raise ValueError(f"No frame with |F| > {CONTACT_N} N in {ds_meta.repo_id} to set the F/T scale.")
    scale = np.maximum(np.quantile(np.abs(w[contact]), QUANTILE, axis=0), 1e-3)
    return scale.astype(float).tolist(), int(contact.sum())
