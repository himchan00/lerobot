# LeRobot Copilot instructions

LeRobot is a Python 3.12+ PyTorch library for robot control, episode-based datasets, policy training,
and simulation evaluation. Use `uv` for contributor commands and treat `pyproject.toml` as the source of
truth for dependencies, extras, CLI entry points, Ruff, Bandit, and mypy configuration.

For end-user setup, hardware, recording, policy selection, and training guidance, follow
[`AGENT_GUIDE.md`](../AGENT_GUIDE.md). For changes under
`src/lerobot/policies/latent_sde/`, also read that directory's `CLAUDE.md`; it defines policy-specific
loss, timing, conditioning, and dataloader invariants that should not be duplicated or weakened here.

## Setup, build, test, and quality commands

```bash
# Base environment, or the usual contributor environment
uv sync --locked
uv sync --locked --extra test --extra dev

# Install every optional integration when broad validation is required
uv sync --locked --extra all

# Required by tests that use repository artifacts
git lfs install
git lfs pull

# Release-package build (the release workflow uses python -m build)
uv run --with build python -m build

# Full pytest suite
uv run pytest tests -vv --maxfail=10

# One test file
uv run pytest tests/configs/test_default.py -sv

# One test function
uv run pytest tests/configs/test_default.py::test_dataset_config_valid -sv

# Lint, format, type checks, spelling, Markdown formatting, and security hooks
uv run pre-commit run --all-files
uv run pre-commit run --files path/to/changed_file.py path/to/changed_test.py

# End-to-end train/eval smoke tests; DEVICE may be cpu or cuda
DEVICE=cpu uv run make test-end-to-end
DEVICE=cpu uv run make test-act-ete-train
```

Tests are dependency-tiered. Match the changed subsystem with an extra such as
`uv sync --locked --extra test --extra dataset`, `--extra hardware`, or `--extra viz`; CI runs the
entire suite separately in those tiers before the all-extras jobs. Tests requiring unavailable optional
packages should collect and skip through existing `is_package_available`, `pytest.importorskip`, or
`tests/utils.py` guards. Set `LEROBOT_TEST_DEVICE=cuda` for pytest device selection; the Makefile uses
`DEVICE=cuda`. Video tests require FFmpeg.

## Architecture and execution flow

- **CLI and configuration:** `[project.scripts]` maps `lerobot-*` commands to `src/lerobot/scripts/`.
  Entrypoints use `@lerobot.configs.parser.wrap()` to parse nested dataclass options with `draccus`.
  Polymorphic configs such as `PreTrainedConfig`, `EnvConfig`, `RobotConfig`, `CameraConfig`, and
  `TeleoperatorConfig` inherit `draccus.ChoiceRegistry`; `--policy.type=act` selects a class registered
  with `@PreTrainedConfig.register_subclass("act")`. The parser handles `--policy.path` and similar
  local/Hub checkpoint paths outside the dataclass, then applies CLI overrides. A `*.path` and `*.type`
  for the same field are mutually exclusive.
- **Training:** `lerobot_train.train()` turns `TrainPipelineConfig` into train/eval datasets, a policy or
  reward model, processor pipelines, optimizer/scheduler, and an Accelerate training loop. Dataset
  metadata supplies feature shapes, FPS, and normalization statistics. Checkpoints combine the policy
  config and `model.safetensors`, serialized pre/postprocessors, `train_config.json`, and resumable
  optimizer/scheduler/RNG state.
- **Datasets:** `LeRobotDatasetMetadata` describes the v3 dataset contract; `LeRobotDataset` coordinates
  `DatasetReader` and `DatasetWriter`. Tabular values and episode metadata are Parquet, visual
  observations are MP4 or images, and the Hub is the normal distribution layer. Policy
  `{observation,action,reward}_delta_indices` are converted to timestamps using dataset FPS, so changing
  those config properties changes the temporal samples delivered by the dataloader.
- **Policies and processors:** `PreTrainedPolicy` is both `torch.nn.Module` and `HubMixin`. Policies do
  not own all input/output adaptation: serializable `PolicyProcessorPipeline`s perform renaming,
  batching, device transfer, normalization, tokenization, and action conversion. Pipelines convert
  external values through the shared `EnvTransition` contract and track feature changes with
  `PolicyFeature`/`FeatureType`.
- **Simulation and hardware:** Evaluation constructs Gymnasium vector environments from `EnvConfig`,
  then applies environment processors and policy pre/postprocessors around rollouts. Recording is a
  separate teleoperation path: `Robot` observation -> robot observation processor -> `Teleoperator`
  action -> teleop/robot action processors -> `Robot.send_action()` -> `LeRobotDataset`. Learned-policy
  deployment belongs in `lerobot-rollout`, not `lerobot-record`.

## Repository-specific conventions

- Registrations happen at import time. A new config subclass is not available to draccus merely because
  its file exists; ensure the relevant package or CLI imports its registration module before parsing.
  Installed third-party packages prefixed with `lerobot_robot_`, `lerobot_camera_`,
  `lerobot_teleoperator_`, `lerobot_policy_`, or `lerobot_env_` are imported by
  `register_third_party_plugins()`. Explicit `--*.discover_packages_path=<module>` loading is also
  supported.
- A built-in policy normally has `configuration_<name>.py`, `modeling_<name>.py`, and
  `processor_<name>.py`. Its config registers a choice and defines validation, optimizer/scheduler
  presets, and temporal delta indices. Its policy sets `config_class` and `name` and implements
  `forward`, `select_action`, `predict_action_chunk`, `reset`, and `get_optim_params`. Its processor
  module exposes `make_<name>_pre_post_processors`. Wire built-ins through the policy package exports
  and all relevant branches in `policies/factory.py`, plus the matching optional extra, tests, and
  policy documentation.
- Keep policy model imports lazy. `lerobot.policies.__init__` intentionally exports lightweight config
  classes but not modeling classes because many policies have heavy optional dependencies. Guard
  optional integrations with the existing import utilities or import them inside factories/functions;
  a base install must still import and collect tests.
- Third-party policy discovery depends on naming: `ThingConfig` in `configuration_thing.py`,
  `ThingPolicy` in `modeling_thing.py`, and `make_thing_pre_post_processors` in
  `processor_thing.py`. Preserve those names when relying on dynamic factories.
- Use canonical feature keys from `lerobot.utils.constants`, including `observation.state`,
  `observation.environment_state`, `observation.image`/`observation.images.<camera>`, `action`, and
  `next.*`. Feature dictionaries, processor `transform_features()`, dataset schemas, and actual
  observation/action payloads must agree; use existing feature and converter utilities instead of
  ad-hoc key translation.
- A new `ProcessorStep` must implement both data transformation and `transform_features()`. Register
  serializable steps with a stable `ProcessorStepRegistry` name and implement `get_config()` plus
  `state_dict()`/`load_state_dict()` when stateful. Registry names are persisted in checkpoints, so do
  not rename them without a compatibility migration.
- Robot and teleoperator implementations pair a registered config with a runtime class exposing the
  expected `config_class`/`name`; camera backends register `CameraConfig` subclasses and are built by
  `make_cameras_from_configs()`. Robot/teleoperator feature properties must match the flat dictionaries
  returned or accepted at runtime and must be available before connecting hardware. Reuse the existing
  lazy factories and calibration paths.
- Mypy is gradual. Errors are enforced for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`,
  `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, and `lerobot.transport`; configs additionally
  prohibit incomplete or untyped definitions. Ruff targets Python 3.12 with a 110-character line
  length, and pre-commit is the authoritative combined quality command.
- Significant AI assistance must be disclosed in the pull-request description per `AI_POLICY.md`.
