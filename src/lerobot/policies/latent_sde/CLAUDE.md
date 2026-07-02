# LatentSDE policy — implementation reference

Latent-SDE policy for hierarchical manipulation (`research_brief.md` is the design source).
Built as a **like-for-like swap of DiffusionPolicy's denoising U-Net**: same vision backbone
(`DiffusionRgbEncoder`), same FiLM-with-scale conditioning, same `GroupNorm` / `down_dims`
ladder. The only structural difference: the chunk-horizon Conv1d collapses to a **point-wise
Linear** because the SDE is integrated one step at a time on the measured state.

## Core idea

Per tick, a drift net predicts an action via one Euler–Maruyama step:
`a ≈ x + μ(x_aug, [h,z])·dt + σ_eff·√dt·ε`. Training is a **β-VAE**: reconstruct the demo
action's one-step velocity, regularized by `KL[q‖p]` on a per-chunk latent strategy `z`.

## The three quantities (clocks differ)

| sym | meaning | clock | fed to net as |
|-----|---------|-------|---------------|
| `x` | measured robot state (proprio). PushT: 2-D `observation.state`. | every tick (fast) | **augmented** = `n_obs_steps` frames flattened → `(B, n_obs·state_dim)`; residual anchored to the most recent frame (`x_now = x_aug[..., -state_dim:]`) |
| `h` | perception conditioning. **Image-only** in this PoC: vision encoder over `n_obs_steps` frames, flattened. | refreshed every `n_action_steps` ticks (matches DP vision duty cycle) | FiLM cond |
| `z` | per-chunk latent strategy (CVAE). | re-sampled every `n_action_steps` ticks with each `h` refresh | concatenated into the net **input**: `net_in = [x_aug, z]` (FiLM cond is `h` only) |

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
- `_TrajEncoder` — **TCN** (dilated Temporal Conv Net, Bai et al. 2018) over a variable-length trajectory: 1×1 channel lift + kernel-3 dilated residual blocks (`_TCNResidualBlock`, dilation 1,2,4,… doubling per level) + masked mean-pool. `RF = 1 + 4·(2^L − 1)` grows exponentially with depth. Depth: **3 levels** → RF ≈ 32 ≈ chunk length H (`LatentSDEModel.__init__`). `_MaskedGroupNorm` / `_masked_mean_pool` support it (pads zeroed before every dilated conv → eval outputs bit-equivalent to exact-length, no cross-pad leak).

## Latent z subsystem

`use_latent_z=False` recovers the no-z PoC exactly (no prior/posterior, no KL, `z_dim=0`).

**Prior `p(z|h)`** (inference + KL target). `conditional_prior=True` → MLP `LatentPrior`; `False` → `StandardNormalPrior` (`N(0,I)`). VQ swaps in `LatentPriorVQ` / `LearnableCategoricalPrior`. `conditional_prior` gates `h` into the **whole z subsystem** (one flag `z_uses_h = use_latent_z ∧ conditional_prior`): when `True` both the prior MLP and the posterior take `h`; when `False` both are h-free.

**Posterior `q(z | (x,a)_{0:H}, [h])`** (training only). Encodes the **(state, action)
trajectory** concatenated on the channel axis — `_TrajEncoder` input channels = `state_dim + action_dim`.
Trajectory = `cat([x_seq, action_target])` `(B, H, state_dim + action_dim)`, all-True mask; pooled feats concat the chunk `h` iff `conditional_prior=True` (else h-free). Both channels match the drift target's pairs: `x_seq` is the **noised** chunk anchor and `action_target` its **time-aligned** action.

**Sampling.** Gaussian: reparam `z = μ_q + σ_q·ε`. FSQ (Mentzer 2309.15505; `use_vq=True`):
deterministic `z_e` → bounded scalar grid + straight-through rounding → `z_q` (no learnable
codebook / EMA / commitment loss); the flat categorical prior over `prod(fsq_levels)` codes is
trained by CE on the (detached) flat code index. The prior also gets a **detached `h`** (`h.detach()`)
— a pure observer, so its CE never shapes the vision encoder (2-stage VQ-VAE spirit). The Gaussian
prior keeps `h` attached (standard CVAE joint training).

**Inference uses the prior only** (no future actions available) — `sample_z_from_prior`.
`z` is re-sampled with each `h` refresh (every `n_action_steps` ticks).

## Loss (`LatentSDEModel.compute_loss`)

```
recon = mean‖μ − v*‖²,   v* = (a − x)/dt          # masked .mean(): padded→0, normalized by nominal H·D
KL    = kl_weight/(H·action_dim·dt) · KL[q‖p]      # ELBO-exact under Δx ~ N(μ·dt, σ²·dt)
loss  = recon + KL   (or recon + prior-CE; FSQ has no commit loss)
```

- `kl_weight = 2·σ_eff²` exactly, where `σ_eff = √(kl_weight/2)` is the SDE diffusion coeff (also the inference noise scale). So tuning `kl_weight` sets both the KL strength and the rollout stochasticity.
- `kl_min` = per-dim free bits (clamp each dim's KL ≥ `kl_min` before summing) to fight posterior collapse.
- **Padding-free chunks.** With `H = n_action_steps` and `drop_n_last_frames = n_action_steps − 1`, the last valid anchor's chunk ends exactly at the episode's final frame — no chunk overflows the episode end. So `compute_loss` needs no `action_is_pad` masking: recon is a plain `.mean()` over `(B, H, D)`, and the per-chunk posterior uses an all-True `valid_mask`.



`x_seq` (measured state, no teacher-forcing) drives the recon target and the drift input; the
posterior encodes `concat([state, action])`, with the **state noised to match the drift input**
when `state_noise_std>0`. The posterior keeps **time-aligned actions**.

- **Train-only augmentations (default off).** `state_noise_std>0` perturbs the chunk's measured-state window by `std·√dt` per frame (giving the noised drift input `x_seq`/`x_seq_aug`); the drift recon target keeps the **time-aligned (original) action** — each noised anchor is regressed toward `action_target` (`= (action − noised anchor)/dt`), a corrective drift toward the demo action (off-manifold stability). The **posterior also sees the noised state** (the same `x_seq`) for consistency with the drift input; its action stays time-aligned. `h_dropout_prob>0` replaces `h` with the learnable `null_h` per sample at source (prior/posterior/drift all see it). Both gate on `self.training`; inference and validation are unaffected.

## Invariants & gotchas

- **`action_dim == state_dim` is assumed** — recon is `(action − state)/dt` (kinematic imitation `x_d ≈ x`, exact for PushT where action = next EE-pose target). Swapping the posterior to actions is therefore dimensionally a no-op but semantically correct.
- **`h` requires ≥1 image feature** — `LatentSDEModel.__init__` raises otherwise.
- **Dataloader windows** come from the config properties: state gets `[1−n_obs, H)` (past + future, so `compute_loss` sees the real demo trajectory), images get only the past `n_obs` frames, actions get `[0, H)` anchored at "now" (not shifted like DP). `H = n_action_steps` is the training-chunk length **and** the deploy h-refresh period.
- `drop_n_last_frames` (default `n_action_steps − 1`) drops the last `n_action_steps − 1` anchors of each episode so every training chunk `[anchor, anchor+H)` stays within the episode (padding-free); `compute_loss` therefore uses a plain mean with no `action_is_pad` masking.
- `deterministic_inference=True` → drift-only rollout (no SDE noise). `deterministic_z_inference` → use `μ_p` instead of sampling z (debug/ablation).
- Drift net never learns σ; do not add an action-σ head without revisiting the `kl_weight = 2σ_eff²` identity.

## Trainer hooks

No custom trainer hooks: the policy trains through the stock `lerobot_train.py` loop unchanged.

Factory wiring: `policies/factory.py` (`get_policy_class` / `make_policy_config` / processor factory, all gated on `name == "latent_sde"`).

## Config cheat-sheet (`LatentSDEConfig`)

`n_obs_steps`, `horizon` (= SDE training-chunk length H), `n_action_steps` (= deploy h-refresh period; ≤ horizon) · `sde_dt` (None → 1/fps) ·
`use_latent_z`, `z_dim`, `conditional_prior` (gates h into prior **and** posterior) ·
`kl_weight`, `kl_min`, `sigma_activation` (`exp`|`softplus`), `z_sigma_min` ·
`use_vq`, `fsq_levels` (per-dim levels; #codes = prod, z_dim = len), `fsq_prior_weight` ·
`state_noise_std` (train-only drift-input noise; corrective target), `h_dropout_prob` (train-only h→null_h dropout) ·
`deterministic_inference`, `deterministic_z_inference` · vision/optim knobs copied verbatim from `DiffusionConfig` for fairness.
