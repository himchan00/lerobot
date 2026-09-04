# LatentSDE policy — implementation reference

Latent-SDE policy for hierarchical manipulation (`research_brief.md` is the design source).
Built as a **like-for-like swap of DiffusionPolicy's denoising U-Net**: same vision backbone
(`DiffusionRgbEncoder`), same FiLM-with-scale conditioning, same `GroupNorm` / `down_dims`
ladder. The only structural difference: the chunk-horizon Conv1d collapses to a **point-wise
Linear** because the SDE is integrated one step at a time on the measured state.

## Core idea

Per tick, a drift net predicts an action via one step: `a ≈ x + μ(x_aug, [h,z])·dt +
rollout_sigma·√dt·ε` (`rollout_sigma=0` ⇒ deterministic drift-only). Training is a **β-VAE**
regularized by `KL[q‖p]` on a per-chunk latent strategy `z`; the reconstruction has two modes
(`config.recon_mode`, the only `compute_loss` branch): **"sde"** reconstructs the demo action's
one-step velocity (teacher-forced), **"ode"** reconstructs the demo *state path* by rolling the
drift through the env's PD dynamics (free-running). Inference is shared across both.

## The three quantities (clocks differ)

| sym | meaning | clock | fed to net as |
|-----|---------|-------|---------------|
| `x` | measured robot state (proprio). PushT: 2-D `observation.state`. | every tick (fast) | the **CURRENT single frame** `(B, state_dim)` — the drift is velocity-blind; scene/motion context lives in `h`. Residual anchored to this frame. |
| `h` | perception conditioning. **Image-only** in this PoC: vision encoder over `n_obs_steps` frames, flattened. | refreshed every `n_action_steps` ticks (matches DP vision duty cycle) | FiLM cond |
| `z` | per-chunk latent strategy (CVAE). | re-sampled every `n_action_steps` ticks with each `h` refresh | Two independent knobs: `drift_uses_h` puts `h` in the FiLM cond; `z_mode="cond"` puts `z` in the FiLM cond, `z_mode="input"` puts `z` in the net input. cond = concat of the enabled pieces (may be empty → plain MLP). |

`observation.environment_state` is deliberately ignored (keeps the h-is-image-only contract).

## Files

- `configuration_latent_sde.py` — `LatentSDEConfig` (draccus `@register_subclass("latent_sde")`). All knobs + `__post_init__` validation + the three `*_delta_indices` properties that shape the dataloader windows.
- `modeling_latent_sde.py` — everything else (see map below).
- `processor_latent_sde.py` — pre/post pipelines. Identical to DP's (normalize state+image+action / unnormalize action). Own factory so future LatentSDE-specific steps have a home.
- `__init__.py` — exports `LatentSDEConfig`, `LatentSDEPolicy`, `make_latent_sde_pre_post_processors`.

### `modeling_latent_sde.py` map

- `LatentSDEPolicy` — LeRobot policy interface (`reset` / `select_action` / `predict_action_chunk` / `forward`) + DP-style obs queues + the h/z inference cache. Thin wrapper.
- `LatentSDEModel` — assembles `h`, holds prior/posterior/vq + drift net, exposes `compute_loss` (train) and `step` (inference). Mirrors `DiffusionModel`.
- `LatentSDEDriftDiffusionNet` / `FiLMResidualMLPBlock` — point-wise FiLM-ResNet hourglass, port of `DiffusionConditionalUnet1d`. Outputs drift `μ` only (action σ² is an EMA-calibrated buffer, not a net output).
- Latent-z modules: `LatentPrior` (Gaussian p(z|h)), `LatentPriorVQ` (VQ p(k|h)), `LatentPosteriorTraj` (Gaussian), `LatentPosteriorTrajVQ` (deterministic + VQ).
- `_TrajEncoder` — **TCN** (dilated Temporal Conv Net, Bai et al. 2018) over a variable-length trajectory: 1×1 channel lift + kernel-3 dilated residual blocks (`_TCNResidualBlock`, dilation 1,2,4,… doubling per level) + masked mean-pool. `RF = 1 + 4·(2^L − 1)` grows exponentially with depth. Depth: **auto-sized** `num_levels = max(1, round(log2(horizon/4)))` so RF ≈ `horizon` (e.g. horizon 8→1, 16→2, 32→3, 64→4), set in `LatentSDEModel.__init__`. `_MaskedGroupNorm` / `_masked_mean_pool` support it (pads zeroed before every dilated conv → eval outputs bit-equivalent to exact-length, no cross-pad leak).

## Latent z subsystem

`use_latent_z=False` recovers the no-z PoC exactly (no prior/posterior, no KL, `z_dim=0`).

**Prior `p(z|h)`** (inference + KL target). MLP `LatentPrior` (Gaussian); VQ swaps in `LatentPriorVQ`. The prior always conditions on `h`. `prior_uses_state=True` additionally feeds the **n_obs proprio window** (deltas `1-n_obs..0`, the same causal window as the image `h`) through a state encoder (`Linear→Mish` to `prior_hidden = z_prior_hidden_dim or h_dim`, so the low-dim state isn't drowned by `h`) and concats it into the prior input, giving `p(z | h, x_{1-n_obs..0})`; it widens the state dataloader window by `n_obs-1` past frames (no effect when `n_obs_steps==1`) and the prior stays causal (past only), matching inference. **State stays absolute** (not re-origined by `x0`), because `normalize_state=True` strips absolute proprio from the drift and reinjects it via this prior window (config validation warns unless `prior_uses_state=True` whenever `normalize_state=True`). `drift_uses_h=False` additionally drops `h` from the **drift** net, so `h` reaches the field *only* through `z` (an `h→z→field` bottleneck; the drift is velocity-blind, reading the current single state frame).

**Posterior `q(z | (x,a)_{0:H}, [h])`** (training only). Encodes the **(state, action)
trajectory** concatenated on the channel axis — `_TrajEncoder` input channels = `state_dim + action_dim`.
Trajectory = `cat([x_seq, action_target])` `(B, H, state_dim + action_dim)`, all-True mask; pooled feats concat the chunk `h` iff `posterior_uses_h` (else trajectory-only), then a **2-layer MLP trunk → (μ,σ) heads** (same trunk+heads shape as `LatentPrior`; VQ posterior mirrors `LatentPriorVQ`). Both channels match the drift target's pairs: `x_seq` is the **noised** chunk anchor and `action_target` its **time-aligned** action. `posterior_uses_state=False` drops the state channels (`_TrajEncoder` input = `action_dim`, traj = `action_target` alone) — orthogonal to `posterior_uses_h`.

**Sampling.** Gaussian: reparam `z = μ_q + σ_q·ε`. Discrete (`use_vq=True`, flavor set by
`config.quantizer`): deterministic `z_e` → quantizer → `z_q`, with a flat categorical prior over
`num_codes` trained by CE on the (detached) flat code index.
- `"fsq"` (Mentzer 2309.15505): bounded scalar grid + straight-through rounding — no learnable
  codebook / EMA / commitment loss; `z_dim = len(fsq_levels)`, `num_codes = prod(fsq_levels)`.
- `"vq"` (van den Oord 1711.00937): learnable codebook + EMA (`vq_decay`) + commitment loss
  (`vq_commit_weight`); `z_dim` stays `z_dim`, `num_codes = vq_codebook_size`. Keep the codebook
  ≤ batch_size so `kmeans_init` seeds every code.

The prior gets a **detached `h`** (`h.detach()`) in the VQ path — a pure observer, so its CE never
shapes the vision encoder (2-stage VQ-VAE spirit). The Gaussian prior keeps `h` attached (standard
CVAE joint training).

**Inference uses the prior only** (no future actions available) — `sample_z_from_prior`.
`z` is re-sampled with each `h` refresh (every `n_action_steps` ticks).

## Loss (`LatentSDEModel.compute_loss`)

`compute_loss` shares h-encoding, the padding mask, and the z posterior/prior, then branches on
`use_vq`:

```
non-VQ:  loss = nll + beta·KL[q‖p]          # Gaussian ELBO; also the vanilla use_latent_z=False path
         nll  = mean_{H·D} [0.5·log(2πσ²·dt) + dt·(v*−μ)²/(2σ²)],  v* = (a−x)/dt   # sum over H·D, /H·D, batch-mean
         KL   also divided by H·D            # shared 1/(H·D) scale → β keeps its meaning (β=1 = ELBO)
         σ² = action_var (buffer)           # NOT gradient-trained: EMA of the analytic MLE dt·mean‖v*−μ‖² (σ-VAE)
VQ/FSQ:  loss = mean‖μ − v*‖² + other_loss  # MSE-mean recon (σ untrained here)
         other_loss = fsq_prior_weight·prior-CE            ("fsq"; no commit loss)
                    = vq_commit_weight·commit + vq_prior_weight·prior-CE   ("vq")
```

- **Trainable σ + β.** The Gaussian decoder σ is a learned scalar `log_action_sigma`; training it by
  the NLL lets it reach its ML value, which drops the data-term weight ~100× vs. the old frozen
  `√(kl_weight/2)` and lets KL actually compete. `beta` (β-VAE) is the KL coefficient — it **replaces**
  the old `kl_weight` (whose `/(H·D·dt)` scaling coupled the KL weight to σ). Inference SDE noise per
  step = `σ·√dt`; `deterministic_inference=True` (default) skips it.
- A plain-MSE `recon_loss` is logged in every path for the z-usage / prior-leakage diagnostics
  (`z_usage_gap`, `prior_recon_gap`), which compare recon MSE across z choices.
- **Padding & masking.** `drop_n_last_frames` (default `max(0, horizon − n_action_steps − n_obs_steps + 1)`) keeps the EXECUTED region unpadded, but the predicted tail may be copy-padded at episode ends. With `do_mask_loss_for_padding=True`, `compute_loss` builds `valid = ~action_is_pad` `(B, H)` and (a) zeroes padded ticks in the recon MSE (`recon_se * valid.unsqueeze(-1)`, still normalized by nominal `B·H·D`, DP-style) and (b) passes `valid` as the per-chunk posterior `valid_mask`. With `do_mask_loss_for_padding=False` (default) `valid` is all-True — recon is a plain `.mean()` over `(B, H, D)` and the posterior mask is all-True, identical to the legacy behavior.



The posterior encodes `concat([state, action])` over the chunk and keeps **time-aligned actions**.
In `recon_mode="sde"` the state is `x_seq` (noised to match the drift input when `state_noise_std>0`);
in `"ode"` it is the clean demo state (the DART perturbation is applied only at the rollout start).

- **Train-only augmentations (default off).** `state_noise_std>0` — **"sde"**: perturbs the chunk's measured-state window by `std·√dt` per frame (noised drift input `x_seq`); the recon target keeps the **time-aligned (original) action** so each noised anchor is regressed toward `action_target` (corrective drift, off-manifold stability), and the **posterior sees the same noised state**. **"ode"**: DART-style rigid shift of the rollout START position by `N(0, std²)` (normalized units; velocity preserved), with recon targets kept clean. Gates on `self.training`; inference/validation unaffected.

## Invariants & gotchas

- **`action_dim == state_dim` is assumed** — recon is `(action − state)/dt` (kinematic imitation `x_d ≈ x`, exact for PushT where action = next EE-pose target). Swapping the posterior to actions is therefore dimensionally a no-op but semantically correct.
- **`h` requires ≥1 image feature** — `LatentSDEModel.__init__` raises otherwise.
- **Dataloader windows** come from the config properties: state gets `[0, horizon)` (current + future demo states — the drift is velocity-blind, so no past frames), images get the past `n_obs` frames (→ `h`), actions get `[0, horizon)` (the chunk targets). With `prior_uses_state=True` the state window is widened to `[1-n_obs, horizon)` so the prior also sees the past `n_obs` proprio frames (`compute_loss` slices the trailing `H` for the drift/posterior and the leading `n_obs` for the prior). `H = horizon` is the training-chunk length; `n_action_steps ≤ horizon` is the deploy execute/refresh period.
- `drop_n_last_frames` (default `max(0, horizon − n_action_steps − n_obs_steps + 1)`) drops the last anchors of each episode so the EXECUTED region stays within the episode; the predicted tail may be copy-padded and is masked iff `do_mask_loss_for_padding=True` (else included unmasked, DP-style default).
- `rollout_sigma=0` (default) → deterministic drift-only inference; `>0` → SDE Brownian noise `rollout_sigma·√dt·ε`. `deterministic_z_inference` → use `μ_p` instead of sampling z (debug/ablation).
- `recon_mode="ode"` needs `action_dim == state_dim` and physical-unit MIN_MAX stats (from `dataset_stats`, held in the `state_min/max`, `action_min/max` buffers) for the PD rollout; `pd_k_p`/`pd_k_v`/`pd_n_substeps` set the controller. Drift net never learns σ (action noise is `rollout_sigma`).

## Trainer hooks

No custom trainer hooks: the policy trains through the stock `lerobot_train.py` loop unchanged.

Factory wiring: `policies/factory.py` (`get_policy_class` / `make_policy_config` / processor factory, all gated on `name == "latent_sde"`).

## Config cheat-sheet (`LatentSDEConfig`)

`recon_mode` (`"sde"` single-step teacher-forced | `"ode"` free-running PD rollout — the `compute_loss` branch) ·
`rollout_sigma` (inference SDE noise; 0 ⇒ deterministic) · `pd_k_p`/`pd_k_v`/`pd_n_substeps` (ode PD dynamics) ·
`n_obs_steps`, `horizon` (= training-chunk length H), `n_action_steps` (= deploy h-refresh period; ≤ horizon) · `sde_dt` (None → 1/fps) ·
`use_latent_z`, `z_dim`, `posterior_uses_h` (q reads h too; else trajectory-only), `posterior_uses_state` (q's traj encoder reads state+action; else actions only), `prior_uses_state` (prior reads the n_obs proprio window too; else image-h only), `drift_uses_h` (False ⇒ drift reads `z` only, no `h`; the velocity-blind field is purely z-conditioned) ·
`beta` (β-VAE KL coefficient; replaces `kl_weight`), `sigma_activation` (`exp`|`softplus`), `z_sigma_min` ·
`use_vq`, `quantizer` (`"fsq"` | `"vq"`) · FSQ: `fsq_levels` (per-dim levels; #codes = prod, z_dim = len), `fsq_prior_weight` · VQ: `vq_codebook_size` (#codes; ≤ batch), `vq_commit_weight`, `vq_decay`, `vq_prior_weight` (z_dim = configured z_dim) ·
`state_noise_std` (train-only; "sde" = drift-window noise / corrective target, "ode" = DART start-perturbation), `do_mask_loss_for_padding` (mask copy-padded chunk ticks in recon + posterior) ·
`normalize_state` (re-origin the drift input + posterior traj state/action to the chunk-initial state x_0 → x_{0:H} become displacements 0..x_H−x_0; velocity target & absolute SDE step stay un-shifted; requires action_dim==state_dim and warns without use_latent_z=True + prior_uses_state=True because the drift loses absolute proprio) ·
`deterministic_z_inference` · vision/optim knobs copied verbatim from `DiffusionConfig` for fairness.
