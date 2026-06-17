# LatentSDE policy — implementation reference

Latent-SDE policy for hierarchical manipulation (`research_brief.md` is the design source).
Built as a **like-for-like swap of DiffusionPolicy's denoising U-Net**: same vision backbone
(`DiffusionRgbEncoder`), same FiLM-with-scale conditioning, same `GroupNorm` / `down_dims`
ladder. The only structural difference: the chunk-horizon Conv1d collapses to a **point-wise
Linear** because the SDE is integrated one step at a time on the measured state.

## Core idea

Per tick, a drift net predicts an action via one Euler–Maruyama step:
`a ≈ x + μ(x_aug, [h,z])·dt + σ_eff·√dt·ε`. Training is a **β-VAE**: reconstruct the demo
action's one-step velocity, regularized by `KL[q‖p]` on a per-episode latent strategy `z`.

## The three quantities (clocks differ)

| sym | meaning | clock | fed to net as |
|-----|---------|-------|---------------|
| `x` | measured robot state (proprio). PushT: 2-D `observation.state`. | every tick (fast) | **augmented** = `n_obs_steps` frames flattened → `(B, n_obs·state_dim)`; residual anchored to the most recent frame (`x_now = x_aug[..., -state_dim:]`) |
| `h` | perception conditioning. **Image-only** in this PoC: vision encoder over `n_obs_steps` frames, flattened. | refreshed every `n_action_steps` ticks (matches DP vision duty cycle) | FiLM cond |
| `z` | per-episode latent strategy (CVAE). | resampled with `h` (`per_chunk`) or once per episode (`per_episode`) | FiLM cond, concatenated: `cond = [h, z]` |

`observation.environment_state` is deliberately ignored (keeps the h-is-image-only contract).

## Files

- `configuration_latent_sde.py` — `LatentSDEConfig` (draccus `@register_subclass("latent_sde")`). All knobs + `__post_init__` validation + the three `*_delta_indices` properties that shape the dataloader windows.
- `modeling_latent_sde.py` — everything else (see map below).
- `processor_latent_sde.py` — pre/post pipelines. Identical to DP's (normalize state+image+action / unnormalize action). Own factory so future LatentSDE-specific steps have a home.
- `__init__.py` — exports `LatentSDEConfig`, `LatentSDEPolicy`, `make_latent_sde_pre_post_processors`.

### `modeling_latent_sde.py` map

- `LatentSDEPolicy` — LeRobot policy interface (`reset` / `select_action` / `predict_action_chunk` / `forward`) + DP-style obs queues + the h/z inference cache. Thin wrapper.
- `LatentSDEModel` — assembles `h`, holds prior/posterior/vq + drift net, exposes `compute_loss` (train) and `step` (inference). Mirrors `DiffusionModel`.
- `LatentSDEDriftDiffusionNet` / `FiLMResidualMLPBlock` — point-wise FiLM-ResNet hourglass, port of `DiffusionConditionalUnet1d`. Outputs drift `μ` only (action σ is **not** learned).
- Latent-z modules: `LatentPrior` / `StandardNormalPrior`, `LatentPriorVQ` / `LearnableCategoricalPrior`, `LatentPosteriorTraj` (Gaussian), `LatentPosteriorTrajVQ` (deterministic + VQ).
- `_TrajEncoder` — masked Conv1d encoder + masked mean-pool over a variable-length trajectory; shared by both posteriors. `_MaskedGroupNorm` / `_masked_mean_pool` support it.

## Latent z subsystem

`use_latent_z=False` recovers the no-z PoC exactly (no prior/posterior, no KL, `z_dim=0`).

**Prior `p(z|h)`** (inference + KL target). `conditional_prior=True` → MLP `LatentPrior`; `False` → `StandardNormalPrior` (`N(0,I)`). VQ swaps in `LatentPriorVQ` / `LearnableCategoricalPrior`.

**Posterior `q(z | (x,a)_{0:T}, [h])`** (training only). Encodes the **(state, action)
trajectory** concatenated on the channel axis — `_TrajEncoder` input channels = `state_dim + action_dim`.
- `per_chunk`: trajectory = `cat([x_seq, action_target])` `(B, H, state_dim + action_dim)`, all-True mask; pooled feats concat the chunk `h`.
- `per_episode`: trajectory = **full-episode** (state, action) traj from a RAM cache (left-aligned, padded + `valid_mask`); h-free (posterior built with `h_dim=0`).

**Sampling.** Gaussian: reparam `z = μ_q + σ_q·ε`. VQ (van den Oord 1711.00937): deterministic
`z_e` → quantizer → `z_q`; categorical prior trained by CE on the (detached) code index. The VQ
prior also gets a **detached `h`** (`h.detach()`) — a pure observer, so its CE never shapes the
vision encoder (2-stage VQ-VAE spirit). The Gaussian prior keeps `h` attached (standard CVAE joint training).

**Inference uses the prior only** (no future actions available) — `sample_z_from_prior`.
`per_chunk` re-samples `z` with each `h` refresh; `per_episode` samples once per `reset()`.

## Loss (`LatentSDEModel.compute_loss`)

```
recon = mean‖μ − v*‖²,   v* = (a − x)/dt          # per-element velocity MSE
KL    = kl_weight/(H·action_dim·dt) · KL[q‖p]      # ELBO-exact under Δx ~ N(μ·dt, σ²·dt)
loss  = recon + KL   (or recon + VQ commit + prior-CE)
```

- `kl_weight = 2·σ_eff²` exactly, where `σ_eff = √(kl_weight/2)` is the SDE diffusion coeff (also the inference noise scale). So tuning `kl_weight` sets both the KL strength and the rollout stochasticity.
- `per_episode` adds a per-sample `×H/T_ep` factor (per-trajectory ELBO).
- `kl_min` = per-dim free bits (clamp each dim's KL ≥ `kl_min` before summing) to fight posterior collapse.

`x_seq` (measured state, no teacher-forcing) still drives the recon target and the drift input;
only the **posterior's** input was switched to actions.

## Invariants & gotchas

- **`action_dim == state_dim` is assumed** — recon is `(action − state)/dt` (kinematic imitation `x_d ≈ x`, exact for PushT where action = next EE-pose target). Swapping the posterior to actions is therefore dimensionally a no-op but semantically correct.
- **`h` requires ≥1 image feature** — `LatentSDEModel.__init__` raises otherwise.
- **`per_episode` needs `set_train_dataset(dataset)`** before training to build the (state, action) trajectory cache (negligible RAM). The trainer calls it via the `hasattr(policy, "set_train_dataset")` hook (`lerobot_train.py`). Action traj is MIN_MAX-normalized via the dataset's `action` stats (matches the normalized `action_target` the `per_chunk` path sees).
- **Dataloader windows** come from the config properties: state gets `[1−n_obs, n_action_steps)` (past + future, so `compute_loss` sees the real demo trajectory), images get only the past `n_obs` frames, actions get `[0, n_action_steps)` anchored at "now" (not shifted like DP).
- `drop_n_last_frames` (default `n_action_steps − 1`) keeps chunks padding-free, so `compute_loss` needs no loss masking.
- `deterministic_inference=True` → drift-only rollout (no SDE noise). `deterministic_z_inference` → use `μ_p` instead of sampling z (debug/ablation).
- Drift net never learns σ; do not add an action-σ head without revisiting the `kl_weight = 2σ_eff²` identity.

## Trainer hooks (in `lerobot_train.py`)

| hook | when | purpose |
|------|------|---------|
| `set_train_dataset(dataset)` | once, after dataset build | pre-cache per-episode (state, action) trajs (`per_episode` only) |

Factory wiring: `policies/factory.py` (`get_policy_class` / `make_policy_config` / processor factory, all gated on `name == "latent_sde"`).

## Config cheat-sheet (`LatentSDEConfig`)

`n_obs_steps`, `n_action_steps` (= SDE chunk length & h-refresh period) · `sde_dt` (None → 1/fps) ·
`use_latent_z`, `z_dim`, `z_sampling_mode` (`per_chunk`|`per_episode`), `conditional_prior` ·
`kl_weight`, `kl_min`, `sigma_activation` (`exp`|`softplus`), `z_sigma_min` ·
`use_vq`, `vq_codebook_size`, `vq_commit_weight`, `vq_decay`, `vq_prior_weight` ·
`deterministic_inference`, `deterministic_z_inference` · vision/optim knobs copied verbatim from `DiffusionConfig` for fairness.
