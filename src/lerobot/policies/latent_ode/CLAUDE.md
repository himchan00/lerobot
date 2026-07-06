# LatentODE policy — implementation reference

Latent-ODE policy for hierarchical manipulation (`research_brief.md` is the design source).
Built as a **like-for-like swap of DiffusionPolicy's denoising U-Net**: same vision backbone
(`DiffusionRgbEncoder`), same FiLM-with-scale conditioning, same `GroupNorm` / `down_dims`
ladder. The only structural difference: the chunk-horizon Conv1d collapses to a **point-wise
Linear** because the ODE is integrated one step at a time on the measured state.

## Core idea

Per tick, a drift net predicts an action deterministically: `a = x + μ(x_aug, [h,z])·dt` (no
diffusion noise — inference is always deterministic). Training is a **β-VAE**: a **free-running
rollout** integrates the drift's own predictions through PushT's exact agent dynamics and matches
the rolled state path to the demo state path, regularized by `KL[q‖p]` on a per-chunk latent
strategy `z`. Rolling the whole path (not per-transition teacher forcing) makes a per-chunk `z`
informative — the identifiability fix for posterior collapse.

## The three quantities (clocks differ)

| sym | meaning | clock | fed to net as |
|-----|---------|-------|---------------|
| `x` | measured robot state (proprio). PushT: 2-D `observation.state`. | every tick (fast) | **augmented** = `n_obs_steps` frames flattened → `(B, n_obs·state_dim)`; residual anchored to the most recent frame (`x_now = x_aug[..., -state_dim:]`) |
| `h` | perception conditioning. **Image-only** in this PoC: vision encoder over `n_obs_steps` frames, flattened. | refreshed every `n_action_steps` ticks (matches DP vision duty cycle) | FiLM cond |
| `z` | per-chunk latent strategy (CVAE). | re-sampled every `n_action_steps` ticks with each `h` refresh | **FiLM cond** alongside `h`: `cond = [h, z]` (net input is `x_aug` only) |

`observation.environment_state` is deliberately ignored (keeps the h-is-image-only contract).

## Files

- `configuration_latent_ode.py` — `LatentODEConfig` (draccus `@register_subclass("latent_ode")`). All knobs + `__post_init__` validation + the three `*_delta_indices` properties that shape the dataloader windows.
- `modeling_latent_ode.py` — everything else (see map below).
- `processor_latent_ode.py` — pre/post pipelines. Identical to DP's (normalize state+image+action / unnormalize action). Own factory so future LatentODE-specific steps have a home.
- `__init__.py` — exports `LatentODEConfig`, `LatentODEPolicy`, `make_latent_ode_pre_post_processors`.

### `modeling_latent_ode.py` map

- `LatentODEPolicy` — LeRobot policy interface (`reset` / `select_action` / `predict_action_chunk` / `forward`) + DP-style obs queues + the h/z inference cache. Thin wrapper; passes `dataset_stats` (from `make_policy` kwargs) into the model.
- `LatentODEModel` — assembles `h`, holds prior/posterior/vq + drift net + the `state_min/max` and `action_min/max` MIN_MAX buffers (populated from `dataset_stats` for the PD rollout's physical units), exposes `compute_loss` (train) and `step` (inference). Mirrors `DiffusionModel`.
- `_pd_rollout` — free-running rollout: integrates the drift's OWN predictions H−1 steps through PushT's exact PD position-controller agent dynamics (pixel space, `pd_n_substeps` substeps per `dt` tick), matched to the demo state path (per-sample MSE). `perturb_init` toggles the DART start-perturbation. `_unnorm`/`_norm` convert normalized ↔ physical units via the stat buffers; `_set_norm_stats` populates them. `step` is a plain deterministic drift step (`x + μ·dt`) — `_sde_step`/`_effective_sigma` were **removed** (no ODE diffusion noise anywhere; `step` takes no `noise`/`generator`).
- `LatentODEDriftNet` / `FiLMResidualMLPBlock` — point-wise FiLM-ResNet hourglass, port of `DiffusionConditionalUnet1d`. Outputs drift `μ` only (action σ is **not** learned).
- Latent-z modules: `LatentPrior` / `StandardNormalPrior`, `LatentPriorVQ` / `LearnableCategoricalPrior`, `LatentPosteriorTraj` (Gaussian), `LatentPosteriorTrajVQ` (deterministic + VQ).
- `_TrajEncoder` — **TCN** (dilated Temporal Conv Net, Bai et al. 2018) over a variable-length trajectory: 1×1 channel lift + kernel-3 dilated residual blocks (`_TCNResidualBlock`, dilation 1,2,4,… doubling per level) + masked mean-pool. `RF = 1 + 4·(2^L − 1)` grows exponentially with depth. Depth: **3 levels** → RF ≈ 32 ≈ chunk length H (`LatentODEModel.__init__`). `_MaskedGroupNorm` / `_masked_mean_pool` support it (pads zeroed before every dilated conv → eval outputs bit-equivalent to exact-length, no cross-pad leak).

## Latent z subsystem

`use_latent_z=False` recovers the no-z PoC exactly (no prior/posterior, no KL, `z_dim=0`).

**Prior `p(z|h)`** (inference + KL target). `conditional_prior=True` → MLP `LatentPrior`; `False` → `StandardNormalPrior` (`N(0,I)`). VQ swaps in `LatentPriorVQ` / `LearnableCategoricalPrior`. `conditional_prior` gates `h` into the **whole z subsystem** (one flag `z_uses_h = use_latent_z ∧ conditional_prior`): when `True` both the prior MLP and the posterior take `h`; when `False` both are h-free.

**Posterior `q(z | (x,a)_{0:H}, [h])`** (training only). Encodes the **(state, action)
trajectory** concatenated on the channel axis — `_TrajEncoder` input channels = `state_dim + action_dim`.
Trajectory = `cat([x_seq, action_target])` `(B, H, state_dim + action_dim)`, all-True mask; pooled feats concat the chunk `h` iff `conditional_prior=True` (else h-free). `x_seq` is the **clean demo** state at each chunk-tick and `action_target` its **time-aligned** action; the rollout's train-only start-perturbation never touches the posterior input.

**Sampling.** Gaussian: reparam `z = μ_q + σ_q·ε`. FSQ (Mentzer 2309.15505; `use_vq=True`):
deterministic `z_e` → bounded scalar grid + straight-through rounding → `z_q` (no learnable
codebook / EMA / commitment loss); the flat categorical prior over `prod(fsq_levels)` codes is
trained by CE on the (detached) flat code index. The prior also gets a **detached `h`** (`h.detach()`)
— a pure observer, so its CE never shapes the vision encoder (2-stage VQ-VAE spirit). The Gaussian
prior keeps `h` attached (standard CVAE joint training).

**Inference uses the prior only** (no future actions available) — `sample_z_from_prior`.
`z` is re-sampled with each `h` refresh (every `n_action_steps` ticks).

## Loss (`LatentODEModel.compute_loss`)

```
recon = mean_B( (1/(H·D)) Σ_k ‖x̂_k − x*_k‖² )     # free-running rollout vs demo state path
KL    = kl_weight/(H·action_dim) · KL[q‖p]         # NO dt factor
loss  = recon + KL   (or recon + prior-CE; FSQ has no commit loss)
```

- **Recon = free-running rollout** (`_pd_rollout`): from the true initial window, roll the drift's OWN predictions H−1 steps through PushT's exact PD agent dynamics and match the rolled state path `x̂` to the demo state path `x*` (per-sample MSE). Deploy-consistent per tick: `â = x̂ + μ·dt` → un-normalize with ACTION stats → PD in pixel space → re-normalize with STATE stats → feed back. Needs physical units, so the model holds `state_min/max`, `action_min/max` buffers (populated from `dataset_stats`) and requires `action_dim == state_dim`.
- **KL scale is ELBO-exact** under the Gaussian state-observation model `N(x̂(z), σ_obs²)`: `kl_weight = 2·σ_obs²` exactly, with `σ_obs` = trajectory observation std. **No dt factor** (unlike a per-step ODE). Tuning `kl_weight` sets the KL strength only — inference carries no noise.
- `kl_min` = per-dim free bits (clamp each dim's KL ≥ `kl_min` before summing) to fight posterior collapse.
- **Padding-free chunks.** With `H = n_action_steps` and `drop_n_last_frames = n_action_steps − 1`, the last valid anchor's chunk ends exactly at the episode's final frame — no chunk overflows the episode end. So `compute_loss` needs no `action_is_pad` masking, and the per-chunk posterior uses an all-True `valid_mask`.

The drift input during the rollout is the **rolled window** (the drift's own last prediction fed
back), NOT the demo state. The posterior encodes the **clean demo** chunk `concat([x_seq, action])`
(`x_seq` = demo state at each chunk-tick); train-time perturbation enters only at the rollout start
(below), so recon targets stay clean.

- **Train-only augmentations.** `state_noise_std>0` (default `0.05`) is a DART-style perturbation of the **rollout START only**: a per-sample Gaussian offset `~N(0, state_noise_std²)` in normalized state units, applied as a **rigid shift** of the initial window (perturbs the start position, preserves the initial velocity — the shared offset cancels in the first-difference). The rollout then runs deterministically and is matched to the **clean** demo path — training recovery back onto the demo from an off-manifold start. `_pd_rollout(perturb_init=self.training)`, so validation/inference are unperturbed. The `z_usage_gap` diagnostic is measured on **clean-init** rollouts (true `z_q` vs a batch-rolled `z`) so the start perturbation doesn't contaminate the gap.

## Invariants & gotchas

- **`action_dim == state_dim` is required** — the rollout integrates the drift on the state and reads the action as the next-state target (kinematic imitation `x_d ≈ x`, exact for PushT where action = next EE-pose target). Enforced in `LatentODEModel.__init__` (raises otherwise).
- **`h` requires ≥1 image feature** — `LatentODEModel.__init__` raises otherwise.
- **Dataloader windows** come from the config properties: state gets `[1−n_obs, H)` (past + future, so `compute_loss` sees the real demo trajectory), images get only the past `n_obs` frames, actions get `[0, H)` anchored at "now" (not shifted like DP). `H = n_action_steps` is the training-chunk length **and** the deploy h-refresh period.
- `drop_n_last_frames` (default `n_action_steps − 1`) drops the last `n_action_steps − 1` anchors of each episode so every training chunk `[anchor, anchor+H)` stays within the episode (padding-free); `compute_loss` therefore uses a plain mean with no `action_is_pad` masking.
- **Inference is always deterministic** (`step`: `x_d = x + μ·dt`, no diffusion noise). `deterministic_z_inference` → use `μ_p` instead of sampling z (debug/ablation).
- Drift net outputs μ only (no learned σ, no injected inference noise). `kl_weight = 2·σ_obs²` is the ELBO-exact KL scale under the Gaussian state-observation model — it is NOT an ODE diffusion coefficient.
- **The rollout net is always `torch.compile`d** (`LatentODEModel.__init__`), decoupled from `compile_model`: it's a boxed compiled view of `self.net` (shared params), so inference stays eager and the checkpoint keeps clean `net.*` keys. `compile_model=True` (default off) *additionally* compiles `self.net` in place for inference (prefixes ckpt keys `net._orig_mod.*`).

## Trainer hooks

No custom trainer hooks: the policy trains through the stock `lerobot_train.py` loop unchanged.

Factory wiring: `policies/factory.py` (`get_policy_class` / `make_policy_config` / processor factory, all gated on `name == "latent_ode"`).

## Config cheat-sheet (`LatentODEConfig`)

`n_obs_steps`, `n_action_steps` (= ODE training-chunk length H **and** deploy h-refresh period) · `dt` (drift-integration Δt; None → 1/fps; the PD substep = `dt/pd_n_substeps`) ·
`pd_k_p`, `pd_k_v`, `pd_n_substeps` (PushT PD position-controller constants for the rollout's agent dynamics; env-specific) ·
`use_latent_z`, `z_dim`, `conditional_prior` (gates h into prior **and** posterior) ·
`kl_weight` (β on KL; ELBO-exact scale = `2·σ_obs²`, no dt factor), `kl_min`, `sigma_activation` (`exp`|`softplus`), `z_sigma_min` ·
`use_vq`, `fsq_levels` (per-dim levels; #codes = prod, z_dim = len), `fsq_prior_weight` ·
`state_noise_std` (train-only DART perturbation of the rollout start; recon targets stay clean) ·
`deterministic_z_inference` · vision/optim knobs copied verbatim from `DiffusionConfig` for fairness.
