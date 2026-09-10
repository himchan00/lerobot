# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare LIBERO without its first-import prompt, optionally checking headless rendering."""

import argparse
import importlib.util
import logging
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ASSETS_REPO_ID = "lerobot/libero-assets"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
EVAL_SUITES = "libero_spatial,libero_object,libero_goal,libero_10"


def prepare_libero() -> Path:
    package = importlib.util.find_spec("libero")
    if package is None or package.origin is None:
        raise ImportError("LIBERO is not installed. Install the existing lerobot[libero] extra first.")
    benchmark_root = Path(package.origin).parent / "libero"
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    config_file = config_dir / "config.yaml"
    # hf-libero's get_assets_path uses this cache independently of config.yaml's assets entry.
    assets_dir = Path.home() / ".cache" / "libero" / "assets"
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(assets_dir),
    }
    existing_config = config_file.exists()
    if existing_config:
        paths = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        if not isinstance(paths, dict):
            raise ValueError(f"Expected a path mapping in {config_file}.")
    for key in ("bddl_files", "init_states"):
        if not isinstance(paths.get(key), str) or not Path(paths[key]).is_dir():
            raise ValueError(
                f"LIBERO {key} path is missing or invalid: {paths.get(key)!r}. "
                "Set LIBERO_CONFIG_PATH to a new directory to initialize paths for this installation."
            )

    snapshot_download(
        repo_id=ASSETS_REPO_ID,
        repo_type="dataset",
        revision=ASSETS_REVISION,
        local_dir=assets_dir,
    )
    if not existing_config:
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(yaml.safe_dump(paths), encoding="utf-8")
    logging.info("LIBERO assets: %s", assets_dir)
    logging.info("LIBERO config: %s%s", config_file, " (preserved)" if existing_config else "")
    return config_file


def check_libero() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    if os.environ["MUJOCO_GL"] in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

    import numpy as np

    from lerobot.envs import close_envs, make_env
    from lerobot.envs.configs import LiberoEnv

    envs = make_env(
        LiberoEnv(
            task=EVAL_SUITES,
            task_ids=[0],
            episode_length=2,
            observation_height=256,
            observation_width=256,
        ),
        n_envs=1,
        use_async_envs=False,
    )
    try:
        for suite, tasks in envs.items():
            for task_id, env in tasks.items():
                env.reset(seed=1000)
                env.step(np.zeros((1, 7), dtype=np.float32))
                frame = env.call("render")[0]
                if frame.shape != (256, 256, 3) or frame.dtype != np.uint8:
                    raise ValueError(f"Unexpected LIBERO render: shape={frame.shape}, dtype={frame.dtype}")
                logging.info("LIBERO reset/step/render ready: %s task %s", suite, task_id)
    finally:
        close_envs(envs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Reset, step and render one task per suite.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare_libero()
    if args.check:
        check_libero()


if __name__ == "__main__":
    main()
