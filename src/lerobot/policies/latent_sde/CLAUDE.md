# LatentSDE policy — implementation reference

Latent-SDE policy for hierarchical manipulation (`research_brief.md` is the design source).
Built as a **like-for-like swap of DiffusionPolicy's denoising U-Net**: the ResNet path uses the
same vision backbone (`DiffusionRgbEncoder`) and FiLM-with-scale conditioning, with the same
`GroupNorm` / `down_dims` ladder. The SmolVLM2 path additionally supports token-KV conditioning.
The FiLM drift replaces chunk-horizon Conv1d with **point-wise Linear** layers. The token-KV drift
uses independent expert queries for each measured state, never attention between supervised ticks.

## Core idea

By default (`sde_geometry="euclidean"`), a drift net predicts an action through one of two representations:

- `target_state` (legacy Push-T): `a ≈ x + μ(x_aug, [h,z])·dt + σ·√dt·ε`
- `delta_state` (relative-action environments): `a ≈ μ(x_aug, [h,z])·dt + σ·√dt·ε`

Training is a **β-VAE** regularized by `KL[q‖p]` on a per-chunk latent strategy `z`. The
teacher-forced velocity target is `(a−x)/dt` for `target_state` and `a/dt` for `delta_state`.
`deterministic_inference=True` removes the `σ·√dt·ε` term.
The opt-in `so3_r3_body` path uses body-local action targets and the same Gaussian/MSE
and EMA machinery. Pose integration is a product retraction; the likelihood is a local
tangent approximation, not a globally normalized manifold transition.

## The three quantities (clocks differ)

| sym | meaning                                                                                                                                                  | clock                                                                 | fed to net as                                                                                                                                                                                                           |
|-----|---------|-------|---------------|
| `x` | measured robot state (proprio). PushT: 2-D `observation.state`; LIBERO: 8-D.                                                                             | every tick (fast)                                                     | the **CURRENT single frame** `(B, state_dim)` — the drift is velocity-blind; scene/motion context lives in `h`. The `target_state` residual is anchored to this frame.                                                  |
| `h` | joint causal observation context: image history plus state history for ResNet, or image/task tokens plus one projected token per causal state frame for SmolVLM2. | refreshed every `n_action_steps` ticks (matches DP vision duty cycle) | FiLM context or token K/V                                                                                                                                                                                               |
| `z` | per-chunk latent strategy (CVAE). | re-sampled every `n_action_steps` ticks with each context refresh | `z_mode="cond"` uses FiLM conditioning or expert-query modulation; `"input"` concatenates z with current state before input projection. |

`observation.environment_state` is deliberately ignored; `observation.state` is always part of the
joint causal context.

## Files

- `configuration_latent_sde.py` — `LatentSDEConfig` (draccus `@register_subclass("latent_sde")`). All knobs, derived defaults, and the `*_delta_indices` properties that shape the dataloader windows.
- `modeling_latent_sde.py` — everything else (see map below).
- `processor_latent_sde.py` — pre/post pipelines. The ResNet path remains identical to DP's; the SmolVLM2 path additionally applies the existing task-newline/tokenizer steps and keeps visual normalization as identity.
- `modeling_smolvlm_context.py` — frozen generic SmolVLM2 context encoder plus trainable `LayerNorm + Linear` projection. It does not construct the SmolVLA action expert.
- `modeling_context.py` — joint observation context: pooled `h`, optional per-layer K/V, and valid-token mask.
- `modeling_token_kv.py` — pointwise drift expert reading SmolVLA-compatible layer-wise K/V.
- `geometry.py` — compact Torch-only body local/retract, continuous state features, controller
  adapters, and nearest-pose distance.
- `__init__.py` — exports `LatentSDEConfig`, `LatentSDEPolicy`, `make_latent_sde_pre_post_processors`.

### `modeling_latent_sde.py` map

- `LatentSDEPolicy` — LeRobot policy interface (`reset` / `select_action` / `predict_action_chunk` / `forward`) + DP-style obs queues + the h/z inference cache. Thin wrapper.
- `LatentSDEModel` — assembles `h`, holds prior/posterior/vq + drift net, exposes `compute_loss` (train) and `step` (inference). Mirrors `DiffusionModel`.
- `LatentSDEDriftDiffusionNet` / `FiLMResidualMLPBlock` — point-wise FiLM-ResNet hourglass, port of `DiffusionConditionalUnet1d`. Outputs drift `μ` only; scalar Euclidean or diagonal body variance is EMA-calibrated, not a net output.
- Latent-z modules: `LatentPrior` (Gaussian p(z|h)), `LatentPriorVQ` (VQ p(k|h)), `LatentPosteriorTraj` (Gaussian), `LatentPosteriorTrajVQ` (deterministic + VQ).
- `_TrajEncoder` — **TCN** (dilated Temporal Conv Net, Bai et al. 2018) over a variable-length trajectory: 1×1 channel lift + kernel-3 dilated residual blocks (`_TCNResidualBlock`, dilation 1,2,4,… doubling per level) + masked mean-pool. `RF = 1 + 4·(2^L − 1)` grows exponentially with depth. Depth: **auto-sized** `num_levels = max(1, round(log2(horizon/4)))` so RF ≈ `horizon` (e.g. horizon 8→1, 16→2, 32→3, 64→4), set in `LatentSDEModel.__init__`. `_MaskedGroupNorm` / `_masked_mean_pool` support it (pads zeroed before every dilated conv → eval outputs bit-equivalent to exact-length, no cross-pad leak).

## Latent z subsystem

`use_latent_z=False` removes prior/posterior and KL (`z_dim=0`); joint observation context is unchanged.

**Prior `p(z|h)`** (inference + KL target). MLP `LatentPrior` (Gaussian); VQ swaps in
`LatentPriorVQ`. The prior consumes `ObservationContext.h`; the FiLM drift also reads this vector,
while the token-KV drift reads the context's per-layer K/V. ResNet concatenates the image-window
embedding and an embedding of the absolute
`n_obs_steps` state window. SmolVLM2 appends one projected token per frame of that same causal state
window, ordered oldest to newest, after the image-task prefix. `drift_uses_h=False` drops direct
context from the drift, so context reaches the field only through `z`.

**Posterior `q(z | a_{0:H})`** (training only). It always encodes only the time-aligned
demonstration action trajectory; observation context and state trajectories are not posterior
inputs. A 2-layer MLP trunk produces the Gaussian heads, and the VQ posterior mirrors the same
action-only trajectory contract.

**Sampling.** Gaussian: reparam `z = μ_q + σ_q·ε`. Discrete (`use_vq=True`, flavor set by
`config.quantizer`): deterministic `z_e` → quantizer → `z_q`, with a flat categorical prior over
`num_codes` trained by CE on the detached flat code index.

- `"fsq"` (Mentzer 2309.15505): bounded scalar grid + straight-through rounding — no learnable
  codebook / EMA / commitment loss; `z_dim = len(fsq_levels)`, `num_codes = prod(fsq_levels)`.
- `"vq"` (van den Oord 1711.00937): learnable codebook + EMA (`vq_decay`) + commitment loss
  (`vq_commit_weight`); `z_dim` stays `z_dim`, `num_codes = vq_codebook_size`. Keep the codebook
  ≤ batch_size so `kmeans_init` seeds every code.

The VQ/FSQ prior receives attached context. Prior cross-entropy therefore trains shared trainable
context projections, while frozen backbone parameters remain frozen.

**Inference uses the prior only** (no future actions available) — `sample_z_from_prior`.
`z` is re-sampled with each `h` refresh (every `n_action_steps` ticks).

## Loss (`LatentSDEModel.compute_loss`)

`compute_loss` shares h-encoding, the padding mask, and the z posterior/prior, then branches on
`use_vq`. The following formulas describe the default Euclidean path:

```
non-VQ:  loss = nll + beta·KL[q‖p]          # Gaussian ELBO; also the vanilla use_latent_z=False path
         nll  = mean_{H·D} [0.5·log(2πσ²·dt) + dt·(v*−μ)²/(2σ²)]
         v*   = (a−x)/dt (`target_state`) or a/dt (`delta_state`)
         KL   also divided by H·D            # shared 1/(H·D) scale → β keeps its meaning (β=1 = ELBO)
         σ² = action_var (buffer)           # NOT gradient-trained: EMA of the analytic MLE dt·mean‖v*−μ‖² (σ-VAE)
VQ/FSQ:  loss = mean‖μ − v*‖² + other_loss  # MSE-mean recon (σ untrained here)
         other_loss = fsq_prior_weight·prior-CE            ("fsq"; no commit loss)
                    = vq_commit_weight·commit + vq_prior_weight·prior-CE   ("vq")
```

- **Calibrated σ + β.** The Gaussian decoder variance is the non-gradient `action_var` buffer,
  warm-started and EMA-calibrated toward the analytic batch MLE. `beta` is the β-VAE KL coefficient.
  Inference SDE noise per step is `σ·√dt`; `deterministic_inference=True` (default) skips it.
- A plain-MSE `recon_loss` is logged in every path for the z-usage / prior-leakage diagnostics
  (`z_usage_gap`, `prior_recon_gap`), which compare recon MSE across z choices.
- **Padding & masking.** `drop_n_last_frames` (default `max(0, horizon − n_action_steps − n_obs_steps + 1)`) keeps the EXECUTED region unpadded, but the predicted tail may be copy-padded at episode ends. With `do_mask_loss_for_padding=True`, `compute_loss` builds `valid = ~action_is_pad` `(B, H)` and (a) zeroes padded ticks in the recon MSE (`recon_se * valid.unsqueeze(-1)`, still normalized by nominal `B·H·D`, DP-style) and (b) passes `valid` as the per-chunk posterior `valid_mask`. With `do_mask_loss_for_padding=False` (default) `valid` is all-True — recon is a plain `.mean()` over `(B, H, D)` and the posterior mask is all-True, identical to the legacy behavior.
  The posterior keeps **time-aligned actions** and does not read state or observation context.
  With `normalize_state=True`, its target-state action trajectory is still re-centered on the clean
  chunk-initial state. Delta-state posterior commands remain unchanged, including in pose mode;
  no state vector is subtracted from a delta action.

- **Train-only augmentation.** In `target_state`, `state_noise_std>0` perturbs the
  measured-state window by `std·√dt` per frame and recomputes the corrective target toward the
  time-aligned action. Euclidean `delta_state` requires `state_noise_std=0`; pose mode supports
  the geometry-aware augmentation below.

## Opt-in body-frame product geometry

`sde_geometry="so3_r3_body"` supports LIBERO relative OSC_POSE commands. Require
`action_representation="delta_state"`, explicit finite positive `sde_dt`,
and `STATE=IDENTITY`, `ACTION=IDENTITY` statistical normalization. Visual preprocessing is unchanged.
The caller is responsible for matching the environment and observation layout.

Keep raw state8/action7, the existing temporal windows, and the clean action-only posterior.
Geometry uses flat poses `[p(3), q_xyzw(4)]` and six-dimensional normalized body increments,
but state features remain 11-dimensional: `[p, R[:,0], R[:,1], finger_qpos]`, with no feature scaling.
Joint observation context always uses absolute pose features. With the default
`normalize_state=True`, only the drift query uses the clean chunk-start pose `(p0, R0)` as its frame:

```text
p_relative = R0.T @ (p - p0)
R_relative = R0.T @ R
drift_features = [p_relative, R_relative[:,0], R_relative[:,1], finger_qpos]
```

Finger positions stay absolute. Do not subtract axis-angle vectors or rotation-column features.
The reference is the current state at context refresh, not the oldest observation in the context
window; it is held for `n_action_steps` ticks and cleared on reset. Training uses that same clean
origin even when drift queries are augmented. `normalize_state=False` keeps absolute drift features.
This is a drift-input coordinate change, not full-policy frame invariance: context still contains
absolute proprioception and camera information.

Fixed LIBERO unit conversion (0.05 m / 0.5 rad per command) stays private to `geometry.py`;
it is not a tunable metric or diffusion setting.

```text
local(x,y):   u = [R_x.T @ (p_y-p_x) / 0.05, Log(R_x.T @ R_y) / 0.5]
retract(x,u): p_out = p_x + R_x @ (0.05*u[:3])
              R_out = R_x @ Exp(0.5*u[3:])
```

Training uses normalized body-controller increments: the model converts a clean relative controller
action into a nominal endpoint, recomputes `local(query, target)`, concatenates the independent
gripper command, and divides by `dt`. Body `action_var` stays diagonal per output coordinate, while
Euclidean mode keeps the scalar variance path. Sampling uses one ordinary Gaussian draw in those
same seven output coordinates (six body, one gripper) and clips only the final controller command.
These output coordinates remain local to the current measured pose, not the chunk-start frame;
controller conversion still uses the absolute current pose.

`action_anchor="nearest"` uses only the squared local pose increment norm for selection. State noise
uses the existing schedule and one six-dimensional Gaussian in those same units. Padding is excluded
from nearest selection regardless of loss masking. The likelihood remains a **local tangent
approximation**, not a globally normalized manifold density.

Mathematical background: [Micro Lie Theory](https://arxiv.org/abs/1812.01537) for composite
manifolds and exponential conventions, [Reparameterizing Distributions on Lie Groups](https://proceedings.mlr.press/v89/falorsi19a.html)
for exponential-map change of variables, and [Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763)
for geodesic random-walk SDE discretization. Action adapters and the local approximation above
are this policy's design, not a claim that those papers implement LIBERO imitation.

## Invariants & gotchas

- Euclidean `target_state` requires `action_dim == state_dim` and uses `(action−state)/dt`, preserving the
  Push-T action semantics. `delta_state` permits different dimensions, uses `action/dt`, and emits
  `μ·dt`; the measured state remains the drift input. The Euclidean LIBERO path keeps all seven action
  channels continuous and requires `state_noise_std=0`. `normalize_state=True` subtracts the clean
  chunk-initial state from Euclidean drift inputs without changing delta-state posterior commands.
  `action_anchor` is unused in Euclidean delta mode.
- **`h` requires ≥1 image feature** — environment state does not replace visual context.
- **Dataloader windows** come from the config properties: state always gets
  `[1-n_obs_steps, horizon)`; images get `[1-n_obs_steps, 1)`; actions get `[0, horizon)`.
  The leading `n_obs_steps` states form the causal context, while the trailing `H` states form the
  teacher-forced drift trajectory, overlapping at the current frame. `H = horizon`;
  `n_action_steps ≤ horizon` is the deploy execute/refresh period.
- `drop_n_last_frames` (default `max(0, horizon − n_action_steps − n_obs_steps + 1)`) drops the last anchors of each episode so the EXECUTED region stays within the episode; the predicted tail may be copy-padded and is masked iff `do_mask_loss_for_padding=True` (else included unmasked, DP-style default).
- `deterministic_inference=True` (default) gives drift-only inference; otherwise scalar or diagonal
  `action_var` supplies noise. `deterministic_z_inference` uses `μ_p` instead of sampling
  `z` (debug/ablation).

## Trainer hooks

No custom trainer hooks: the policy trains through the stock `lerobot_train.py` loop unchanged.

Factory wiring: `policies/factory.py` (`get_policy_class` / `make_policy_config` / processor factory, all gated on `name == "latent_sde"`).

## Config cheat-sheet (`LatentSDEConfig`)

Policy-specific compatibility and input preflight checks are omitted. `__post_init__` only derives
`drop_n_last_frames`, FSQ `z_dim`, and ResNet `crop_shape`; `validate_features()` is a no-op required
by the base config interface. The settings and tensor layouts below are caller preconditions, not
automatically enforced compatibility guarantees. Stable rotation operations and valid nearest
candidates remain algorithm requirements. Space selection rejects obsolete or unknown geometry modes.

`context_encoder` (`"resnet"` | `"smolvlm2"`) · `conditioning` (`"film"` | `"token_kv"`;
token-KV requires SmolVLM2) · `action_representation` (`"target_state"` |
`"delta_state"`) · SmolVLM2: `vlm_model_name`, immutable `vlm_model_revision`, `vlm_num_layers`,
`vlm_context_dim`, `vlm_resize_shape`, `tokenizer_max_length` ·
`sde_geometry` (`"euclidean"` | `"so3_r3_body"`) ·
`n_obs_steps`, `horizon` (= training-chunk length H), `n_action_steps` (= deploy h-refresh period; ≤ horizon) · `sde_dt` (default 0.1; Euclidean None falls back to 1.0, not dataset FPS; pose mode requires an explicit value) ·
`use_latent_z`, `z_dim`, action-trajectory-only posterior, joint-context prior,
`drift_uses_h` (False ⇒ drift reads `z` only, no direct `h`) ·
`beta` (β-VAE KL coefficient; replaces `kl_weight`), `sigma_activation` (`exp`|`softplus`), `z_sigma_min` ·
`use_vq`, `quantizer` (`"fsq"` | `"vq"`) · FSQ: `fsq_levels` (per-dim levels; #codes = prod, z_dim = len), `fsq_prior_weight` · VQ: `vq_codebook_size` (#codes; ≤ batch), `vq_commit_weight`, `vq_decay`, `vq_prior_weight` (z_dim = configured z_dim) ·
`state_noise_std` (target-state or pose train-only drift-window noise / corrective target), `do_mask_loss_for_padding` (mask copy-padded chunk ticks in recon + posterior) ·
`normalize_state` (clean chunk-start-relative drift inputs: subtraction in Euclidean mode,
reference-frame pose features in body mode; only target-state posterior actions subtract x_0;
velocity targets, integration, and joint observation context remain unchanged) ·
`deterministic_z_inference` · vision/optim knobs copied verbatim from `DiffusionConfig` for fairness.

## SmolVLM2 / LIBERO usage

The backbone is the pure generic SmolVLM2 checkpoint, not `lerobot/smolvla_base`. Images must be
square; the default pinned model requires `vlm_resize_shape=(512, 512)`. The tokenizer is deliberately
not revision-pinned. Saved policies still need the generic model and tokenizer available locally or
through the Hub when constructing the encoder.

The adapter matches SmolVLA's inference-prefill mathematics: BF16 loading, embedding scaling,
RoPE theta 10000, eager attention, and B-sized vision calls per frame/camera. Its prefix includes
`n_obs_steps` projected state tokens, ordered oldest to newest. Image/task queries cannot attend to
state tokens; each state query can attend to the image/task prefix, itself, and earlier state tokens.
Only the leading causal state window is encoded, never the future teacher-forced states.
`conditioning="film"` pools the joint prefix; `"token_kv"` exposes its token K/V representation to
the compatible head. The `n_obs_steps=1` layout and parameter shapes are unchanged; multi-frame
checkpoints from the earlier current-state-only adapter are not behavior-equivalent.
See the measured parity boundary, simple default-setting commands, and paper settings in
`smolvlm2_head_baselines.md`.

Both context encoders take cameras in `input_features` visual-feature insertion order, like
SmolVLA. There is no separate camera-order option.

The four former posterior/prior conditioning options are removed rather than deprecated. Configs
containing `posterior_uses_h`, `posterior_uses_state`, `prior_uses_state`, or `posterior_state` fail
schema parsing. Existing weights are not guaranteed compatible with the new joint-context modules.

Explicit BF16 training uses `ACCELERATE_MIXED_PRECISION=bf16`; `policy.use_amp` alone does not
configure this trainer's Accelerator. Standalone evaluation also needs its CUDA autocast default
set to BF16 when matching that precision choice. Delta integration preserves the measured-state dtype so
BF16 drift outputs still produce float32 controller actions usable by NumPy.

The token-KV expert inherits the loaded text config's BF16 dtype, while its newly initialized
projections and residual stream can remain FP32. Cast activations to the consuming projection or
MLP's weight dtype at each boundary, including cached latent `z`. Training and inference must work
without autocast; do not force AMP or upcast the whole expert to work around a dtype mismatch.

### Local integration smoke

Install the existing extras; `diffusion` supplies the current Latent-SDE scheduler:

```bash
conda run -n lerobot env CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  uv sync --locked --extra latent_sde --extra training --extra diffusion --extra dataset --extra libero
```

On a fresh LIBERO installation, initialize its paths and download its assets with networking enabled.
Answer `N` to the first-import custom-dataset-path prompt to accept the defaults:

```bash
conda run --no-capture-output -n lerobot uv run python -c \
  'from libero.libero.utils.download_utils import download_assets_from_huggingface; download_assets_from_huggingface()'
```

The following reproduces the integration recipe: one update on pinned dataset episode 0, followed by
two simulator steps. Use fresh output directories. This is not an overfitting experiment or a benchmark
score; the evaluation task is intentionally only an independent simulator-I/O check.

```bash
conda run -n lerobot uv run lerobot-train \
  --policy.type=latent_sde \
  --policy.context_encoder=smolvlm2 \
  --policy.conditioning=token_kv \
  --policy.action_representation=delta_state \
  --policy.normalization_mapping='{"ACTION":"MIN_MAX","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.action_anchor=clean \
  --policy.state_noise_std=0 --policy.sde_dt=0.1 \
  --policy.n_obs_steps=1 --policy.horizon=4 --policy.n_action_steps=2 \
  --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.revision=86958911c0f959db2bbbdb107eb3e17c5f9c798e \
  --dataset.episodes='[0]' \
  --output_dir=outputs/train/latent_sde_libero_smoke \
  --steps=1 --batch_size=1 --num_workers=0 --persistent_workers=false \
  --save_freq=1 --log_freq=1 --env_eval_freq=0 --wandb.enable=false

conda run -n lerobot env MUJOCO_GL=egl uv run lerobot-eval \
  --policy.path=outputs/train/latent_sde_libero_smoke/checkpoints/000001/pretrained_model \
  --policy.device=cuda \
  --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
  --env.observation_height=256 --env.observation_width=256 --env.episode_length=2 \
  --eval.batch_size=1 --eval.n_episodes=1 --eval.use_async_envs=false \
  --output_dir=outputs/eval/latent_sde_libero_smoke
```

The smoke intentionally retains its original `ACTION=MIN_MAX` setting. For a comparison using
SmolVLA's default normalization, explicitly set both policies to `VISUAL=IDENTITY`,
`STATE=MEAN_STD`, `ACTION=MEAN_STD` with the same statistics. Also align checkpoint artifacts and
precision explicitly; the report's basic commands intentionally keep each policy's own presets.

The dataset metadata reports 10 FPS, which motivated the explicit `sde_dt=0.1` in this recipe.
The installed LIBERO controller instead defaults to `control_freq=20`; the environment factory does
not override it. Neither `sde_dt` nor `--env.fps` changes that simulator control frequency. Treat the
current delta output as a normalized controller command, not a verified physical state displacement.
