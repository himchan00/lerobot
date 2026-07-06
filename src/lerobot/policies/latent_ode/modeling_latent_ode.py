#!/usr/bin/env python
#
# LatentODE Policy — point-wise drift network.
#
# Port of DiffusionConditionalUnet1d to the ODE setting: horizon-axis Conv1d → Linear,
# diffusion-timestep encoder removed (no denoising loop), U-Net skip-connections collapse
# into per-block residuals. FiLM-with-scale / GroupNorm / Mish / down_dims widths / FiLM
# conditioning are preserved verbatim for architectural fairness vs. DiffusionPolicy.

import math
import warnings
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..diffusion.modeling_diffusion import DiffusionRgbEncoder, DiffusionSinusoidalPosEmb
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_latent_ode import LatentODEConfig


def _sigma_act(activation: str):
    """σ = act(s) + sigma_min; caller adds the floor. Used by z prior/posterior heads."""
    if activation == "exp":
        return torch.exp
    if activation == "softplus":
        return F.softplus
    raise ValueError(f"sigma_activation must be 'exp' or 'softplus'; got {activation!r}.")


def _maybe_compile(module: nn.Module, mode: str) -> nn.Module:
    """torch.compile with an eager fallback if a compiler backend is unavailable."""
    try:
        return torch.compile(module, mode=mode)
    except Exception as e:  # pragma: no cover - environment-dependent
        warnings.warn(f"torch.compile unavailable ({e!r}); running eager.", stacklevel=2)
        return module


class LatentODEPolicy(PreTrainedPolicy):
    """LatentODE policy (free-space, no compliance) — first PoC.

    Wraps `LatentODEModel` and implements the LeRobot policy interface
    (reset / select_action / forward) with DiffusionPolicy-style observation queues.

    Inference duty cycle (research_brief.md §1.2): h (image conditioning) and z
    (per-episode latent) are refreshed together every `n_action_steps` ticks —
    matching DP's vision-encoder cadence. The drift net runs every tick
    on the current measured state.
    """

    config_class = LatentODEConfig
    name = "latent_ode"

    def __init__(self, config: LatentODEConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self._cached_h: Tensor | None = None
        self._steps_until_h_refresh: int = 0
        self._cached_z: Tensor | None = None

        self.model = LatentODEModel(config, dataset_stats=kwargs.get("dataset_stats"))
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """Clear observation queue and the cached image conditioning. Call on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        self._cached_h = None
        self._steps_until_h_refresh = 0
        self._cached_z = None

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        # Single-step ODE policy has no chunk-at-once inference path; rollout is per-tick.
        raise NotImplementedError(
            "LatentODEPolicy does not support predict_action_chunk (chunked / async inference); "
            "use select_action for per-tick rollout."
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """One deterministic drift step per tick with per-tick state-feedback.

        Refresh h every `n_action_steps` ticks and re-sample z alongside it (chunk-local hold).
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        # Slow path: re-encode h and re-sample z every n_action_steps ticks.
        if self._cached_h is None or self._steps_until_h_refresh == 0:
            stacked_images = torch.stack(list(self._queues[OBS_IMAGES]), dim=1)
            self._cached_h = self.model.encode_observations({OBS_IMAGES: stacked_images})
            self._cached_z = self.model.sample_z_from_prior(self._cached_h)
            self._steps_until_h_refresh = self.config.n_action_steps

        # Fast path: flatten the n_obs_steps state window into x_aug so the drift can read
        # local first-differences ≈ velocity (mirrors DP's state branch in the global cond).
        stacked_state = torch.stack(list(self._queues[OBS_STATE]), dim=1)  # (B, n_obs, D)
        x_aug = stacked_state.reshape(stacked_state.shape[0], -1)           # (B, n_obs * D)
        # Ticks since the last h refresh (0 = refresh tick .. n_action_steps-1); the drift's PE index.
        step_idx = self.config.n_action_steps - self._steps_until_h_refresh
        action = self.model.step(x_aug, self._cached_h, self._cached_z, step_idx=step_idx)
        self._steps_until_h_refresh -= 1
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float] | None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss, loss_dict = self.model.compute_loss(batch)
        return loss, loss_dict


class LatentODEModel(nn.Module):
    """Assembles h, holds drift net, exposes sample + compute_loss.

    Mirrors the role of DiffusionModel one-for-one.
    """

    def __init__(self, config: LatentODEConfig, dataset_stats: dict | None = None):
        super().__init__()
        self.config = config

        # h is image-only by design: proprio state lives on the fast clock as the ODE
        # input x (the n_obs window is flattened in `select_action`); env_state is deferred.
        if not self.config.image_features:
            raise ValueError(
                "LatentODEPolicy requires at least one image feature (h = image-only "
                "in this PoC). Got input_features with no observation.image*."
            )

        global_cond_dim = 0
        num_images = len(self.config.image_features)
        if self.config.use_separate_rgb_encoder_per_camera:
            # DiffusionRgbEncoder reads a few fields from the config; LatentODEConfig matches
            # DiffusionConfig's names so no adaptation is needed.
            encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
            self.rgb_encoder = nn.ModuleList(encoders)
            global_cond_dim += encoders[0].feature_dim * num_images
        else:
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += self.rgb_encoder.feature_dim * num_images

        self.h_dim = global_cond_dim * config.n_obs_steps # h is stacked over n_obs_steps
        self.use_latent_z = config.use_latent_z

        # use_latent_z=False → no prior/posterior/vq, z_dim=0. Otherwise:
        #   * self.prior     : (use_vq) × (config.conditional_prior)
        #   * self.posterior : (use_vq) — trajectory encoder; conditions on h iff z_uses_h
        #   * self.vq        : built iff use_vq
        self.use_vq = config.use_vq
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]

        # Physical-unit MIN_MAX stats for the PD rollout (normalization otherwise lives in the
        # processor, so the model never sees physical units). Registered unconditionally so the
        # state_dict is config-invariant; persisted so they survive save/load. Populated from
        # dataset_stats at train-time construction; read by the PD rollout in compute_loss.
        self.register_buffer("state_min", torch.zeros(self.state_dim))
        self.register_buffer("state_max", torch.ones(self.state_dim))
        self.register_buffer("action_min", torch.zeros(self.action_dim))
        self.register_buffer("action_max", torch.ones(self.action_dim))
        if self.state_dim != self.action_dim:
            raise ValueError(
                f"LatentODE requires action_dim == state_dim (the rollout integrates the drift on the "
                f"state and reads the action as the next-state target); got state_dim="
                f"{self.state_dim}, action_dim={self.action_dim}."
            )
        if dataset_stats is not None:
            self._set_norm_stats(dataset_stats)

        # conditional_prior gates the image features h into the *entire* z subsystem: True → the
        # prior is an h-conditioned MLP and the posterior takes h as input; False → both are h-free.
        self.z_uses_h = config.use_latent_z and config.conditional_prior
        if not self.use_latent_z:
            self.z_dim = 0
            self.prior = None
            self.posterior = None
            self.vq = None
        else:
            self.z_dim = config.z_dim
            posterior_hidden = config.z_posterior_hidden_dim or self.h_dim
            prior_hidden = config.z_prior_hidden_dim or self.h_dim

            # TCN posterior depth auto-sized so its receptive field (≈ 4·2^L) matches the chunk
            # length (horizon), kernel 3. E.g. horizon 8→1, 16→2, 32→3, 64→4.
            tcn_levels = max(1, round(math.log2(config.horizon / 4)))

            if self.use_vq:
                # FSQ index space = prod(levels); the categorical prior/CE/perplexity size to this.
                self.num_codes = math.prod(config.fsq_levels)
                self.prior = (
                    LatentPriorVQ(
                        h_dim=self.h_dim,
                        codebook_size=self.num_codes,
                        hidden_dim=prior_hidden,
                    )
                    if self.z_uses_h
                    else LearnableCategoricalPrior(codebook_size=self.num_codes)
                )
                self.posterior = LatentPosteriorTrajVQ(
                    input_dim=self.state_dim + self.action_dim,  # encodes concat([state, action])
                    h_dim=self.h_dim if self.z_uses_h else 0,  # 0 → h-free posterior
                    z_dim=self.z_dim,
                    hidden_dim=posterior_hidden,
                    num_levels=tcn_levels,
                    n_groups=config.n_groups,
                )
                from vector_quantize_pytorch import FSQ
                # FSQ: bounded scalar grid + straight-through rounding. No learnable codebook,
                # no commitment loss, no dead codes — z_dim is fixed to len(levels) (config).
                self.vq = FSQ(levels=config.fsq_levels)
            else:
                self.prior = (
                    LatentPrior(
                        h_dim=self.h_dim,
                        z_dim=self.z_dim,
                        hidden_dim=prior_hidden,
                        sigma_activation=config.sigma_activation,
                        sigma_min=config.z_sigma_min,
                    )
                    if self.z_uses_h
                    else StandardNormalPrior(z_dim=self.z_dim)
                )
                self.posterior = LatentPosteriorTraj(
                    input_dim=self.state_dim + self.action_dim,  # encodes concat([state, action])
                    h_dim=self.h_dim if self.z_uses_h else 0,  # 0 → h-free posterior
                    z_dim=self.z_dim,
                    hidden_dim=posterior_hidden,
                    sigma_activation=config.sigma_activation,
                    sigma_min=config.z_sigma_min,
                    num_levels=tcn_levels,
                    n_groups=config.n_groups,
                )
                self.vq = None

        # Net input = augmented state (n_obs_steps frames flattened). Cond = concat([h, z]) (FiLM);
        # z conditions the drift alongside h (widens cond_dim by z_dim, 0 if no z), not as a raw input.
        self.net = LatentODEDriftNet(
            input_dim=self.state_dim * config.n_obs_steps,
            action_dim=config.action_feature.shape[0],
            cond_dim=self.h_dim + self.z_dim,
            down_dims=config.down_dims,
            n_groups=config.n_groups,
            use_film_scale_modulation=config.use_film_scale_modulation,
            use_pe=config.use_pe,
            pe_dim=config.pe_dim,
        )

        # The rollout calls the net H−1× per forward (dominant train cost), so ALWAYS compile it — via
        # a boxed view sharing self.net's params, so self.net stays eager (eager inference, clean `net.*`
        # checkpoint keys). Boxed in a list so nn.Module doesn't register it as a duplicate submodule.
        # compile_model additionally compiles self.net in place for inference (opt-in).
        if config.compile_model:
            self.net = _maybe_compile(self.net, config.compile_mode)  # inference too (opt-in)
            self._rollout_net = [self.net]
        else:
            self._rollout_net = [_maybe_compile(self.net, config.compile_mode)]  # rollout only

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute h, the image-only global conditioning vector.

        Slow path: runs the vision encoder(s) over `n_obs_steps` stacked frames. At deployment,
        the result is cached and reused for `n_action_steps` ticks (matches DP's vision-encoder
        duty cycle).

        Returns: (B, num_images * feature_dim * n_obs_steps).
        """
        batch_size, n_obs_steps = batch[OBS_IMAGES].shape[:2]

        if self.config.use_separate_rgb_encoder_per_camera:
            images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
            img_features_list = torch.cat(
                [enc(imgs) for enc, imgs in zip(self.rgb_encoder, images_per_camera, strict=True)]
            )
            img_features = einops.rearrange(
                img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
        else:
            img_features = self.rgb_encoder(
                einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
            )
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )

        return img_features.flatten(start_dim=1)

    def sample_z_from_prior(
        self,
        h: Tensor,
        deterministic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor | None:
        """Sample z ~ p(z|h) (Gaussian or VQ-categorical). Returns None when use_latent_z=False."""
        if not self.use_latent_z:
            return None
        if deterministic is None:
            deterministic = self.config.deterministic_z_inference
        if self.use_vq:
            logits = self.prior(h)
            if deterministic:
                k = logits.argmax(dim=-1)
            else:
                # `torch.multinomial` is the only categorical sampler that accepts `generator`.
                probs = F.softmax(logits, dim=-1)
                k = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            return self.vq.indices_to_codes(k)
        else:
            mu_p, sigma_p = self.prior(h)
            if deterministic:
                return mu_p
            eps = torch.randn(mu_p.shape, dtype=mu_p.dtype, device=mu_p.device, generator=generator)
            return mu_p + sigma_p * eps

    def step(
        self,
        x_aug: Tensor,
        h: Tensor,
        z: Tensor | None,
        step_idx: int = 0,
    ) -> Tensor:
        """One deterministic drift step from pre-computed h (and optional z): x_d = x_now + μ·dt.

        Args:
            x_aug: (B, n_obs_steps * state_dim) — augmented state window. The drift net reads the
                   full window; the update is anchored to the most recent frame (x_aug[..., -state_dim:]).
            step_idx: ticks since the current h refresh (0..n_action_steps-1); the PE index when use_pe.
        """
        cond = torch.cat([h, z], dim=-1) if self.use_latent_z else h
        step = (
            torch.full((x_aug.shape[0],), float(step_idx), dtype=x_aug.dtype, device=x_aug.device)
            if self.config.use_pe
            else None
        )
        mu = self.net(x_aug, cond, step)
        dt = self.config.dt if self.config.dt is not None else 1.0
        return x_aug[..., -self.state_dim:] + mu * dt

    def _set_norm_stats(self, dataset_stats: dict) -> None:
        """Populate the state/action MIN_MAX buffers from dataset stats (physical units for the PD rollout)."""

        def _get(key: str, stat: str, dim: int):
            t = dataset_stats.get(key, {}).get(stat) if dataset_stats else None
            return None if t is None else torch.as_tensor(t, dtype=torch.float32).reshape(-1)[:dim]

        sm, s_max = _get(OBS_STATE, "min", self.state_dim), _get(OBS_STATE, "max", self.state_dim)
        am, a_max = _get(ACTION, "min", self.action_dim), _get(ACTION, "max", self.action_dim)
        if sm is not None and s_max is not None:
            self.state_min.copy_(sm)
            self.state_max.copy_(s_max)
        if am is not None and a_max is not None:
            self.action_min.copy_(am)
            self.action_max.copy_(a_max)

    def _unnorm(self, x_n: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        """MIN_MAX inverse: [-1, 1] → physical units."""
        lo, hi = lo.to(x_n), hi.to(x_n)
        return (x_n + 1.0) / 2.0 * (hi - lo) + lo

    def _norm(self, x_p: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        """MIN_MAX forward: physical units → [-1, 1]."""
        lo, hi = lo.to(x_p), hi.to(x_p)
        return 2.0 * (x_p - lo) / (hi - lo).clamp_min(1e-8) - 1.0

    def _pd_rollout(
        self,
        state_full: Tensor,
        h: Tensor,
        z: Tensor | None,
        perturb_init: bool = False,
        recon_mask: Tensor | None = None,
    ) -> Tensor:
        """Free-running rollout of the drift field through the env's exact agent dynamics.

        Deploy-consistent: the drift emits a normalized action â = x̂ + μ·dt (as in `step`), which is
        un-normalized with ACTION stats, driven through PushT's PD position controller in pixel space
        (kinematic pusher: `v ← v + (k_p·(a−p) − k_v·v)·dt_sub`, `p ← p + v·dt_sub`, `pd_n_substeps`
        substeps per tick), then re-normalized with STATE stats and fed back in. The pusher is a
        kinematic body, so no contact force acts on it and this is the complete, exact agent dynamics.

        Rolls horizon-1 steps from the initial window; returns per-sample MSE (B,) of the rolled state
        path vs the CLEAN demo state path. Full BPTT over the chunk (truncation deferred).

        perturb_init: DART-style train aug — shift the rollout START position by a per-sample Gaussian
        offset ~N(0, state_noise_std²) in normalized units (velocity preserved); targets stay clean.
        recon_mask: (B, horizon-1) bool/float; when given, copy-padded chunk ticks (False) are zeroed
        out of the per-sample sum (DP-style). 
        """
        cfg = self.config
        B = state_full.shape[0]
        n_obs, H = cfg.n_obs_steps, cfg.horizon
        dt = cfg.dt if cfg.dt is not None else 1.0
        sub_dt = dt / cfg.pd_n_substeps
        k_p, k_v = cfg.pd_k_p, cfg.pd_k_v
        s_lo, s_hi, a_lo, a_hi = self.state_min, self.state_max, self.action_min, self.action_max

        # Normalized-state window (last n_obs frames), initialized from the true past observation window.
        window = [state_full[:, i] for i in range(n_obs)]
        if perturb_init and cfg.state_noise_std > 0:
            # Rigid shift of the whole window → perturbs the START position, preserves the initial
            # velocity (a shared offset cancels in the first-difference). Targets stay clean below.
            eta = cfg.state_noise_std * torch.randn(
                B, self.state_dim, dtype=state_full.dtype, device=state_full.device
            )
            window = [w + eta for w in window]
        # Pixel-space (pos, vel) of the current anchor; velocity from the demo first-difference.
        p_px = self._unnorm(window[-1], s_lo, s_hi)
        v_px = (
            (p_px - self._unnorm(window[-2], s_lo, s_hi)) / dt if n_obs >= 2 else torch.zeros_like(p_px)
        )

        rollout_net = self._rollout_net[0]  # compiled net view (see __init__)

        se = state_full.new_zeros(B)
        for k in range(H - 1):
            x_aug = torch.cat(window, dim=-1)
            step = torch.full((B,), float(k), dtype=h.dtype, device=h.device) if cfg.use_pe else None
            cond = torch.cat([h, z], dim=-1) if self.use_latent_z else h
            mu = rollout_net(x_aug, cond, step)
            a_px = self._unnorm(window[-1] + mu * dt, a_lo, a_hi)  # deploy: action = x̂ + μ·dt
            for _ in range(cfg.pd_n_substeps):
                v_px = v_px + (k_p * (a_px - p_px) - k_v * v_px) * sub_dt
                p_px = p_px + v_px * sub_dt
            x_next = self._norm(p_px, s_lo, s_hi)  # back to normalized state (= new window[-1])
            sq = ((x_next - state_full[:, n_obs + k]) ** 2).sum(dim=-1)  # vs demo next state
            if recon_mask is not None:
                sq = sq * recon_mask[:, k].to(sq.dtype)  # zero out copy-padded chunk ticks (DP-style)
            se = se + sq
            window = window[1:] + [x_next]
        return se / (H * self.action_dim)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """β-VAE-style loss: free-running-rollout recon + kl_loss_weight · KL[q‖p].

        Recon: integrate the drift's own predictions through the env's exact agent dynamics
        (`_pd_rollout`) from the true initial window and match the rolled state path to the demo
        state path (per-sample mean squared error). A per-chunk z must bend the whole path onto the
        demo, so it stays informative (identifiability fix for posterior collapse). The generative
        model is a Gaussian observation N(x̂(z), σ_obs²) on each state, so the KL scale is ELBO-exact
        at kl_weight/(H·D) (NO dt) with kl_weight = 2·σ_obs² (σ_obs = trajectory observation std).

        The posterior q(z | demo (x, a) chunk) sees the clean demo. Train-time perturbation enters only
        at the rollout START (DART-style, `state_noise_std`); recon targets stay clean.

        Expected `batch` (normalized + on device; LatentODEPolicy.forward stacks images):
            "observation.state":  (B, n_obs_steps + horizon - 1, state_dim) — past + future
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
            "action":             (B, horizon, action_dim)
            "action_is_pad":      (B, horizon) — used iff do_mask_loss_for_padding

        Padding: `drop_n_last_frames` keeps the EXECUTED region unpadded; the predicted tail may be
        copy-padded at episode ends and is masked (recon + posterior) when do_mask_loss_for_padding.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch

        action_target = batch[ACTION]                        # (B, horizon, action_dim)
        B, H, action_dim = action_target.shape               # H == horizon (prediction chunk length)
        n_obs = self.config.n_obs_steps
        assert self.config.horizon == H, (
            f"compute_loss expects horizon action targets; got H={H} "
            f"vs config.horizon={self.config.horizon}."
        )

        # state_full: (B, n_obs + horizon - 1, state_dim). x_seq[k] = demo state at chunk-tick k.
        state_full = batch[OBS_STATE]
        expected_T = n_obs + H - 1
        if state_full.shape[1] != expected_T:
            raise ValueError(
                f"Expected batch[OBS_STATE] T={expected_T} (n_obs_steps + horizon - 1); "
                f"got T={state_full.shape[1]}. Check observation_delta_indices_per_key."
            )

        # Padding mask (DP-style). `action_is_pad` (B, horizon) marks copy-padded chunk ticks at
        # episode ends; it aligns with action/state deltas 0..horizon-1. A rolled tick k (0..horizon-2)
        # targets the demo state at delta k+1, so its recon mask is valid[:, k+1].
        if self.config.do_mask_loss_for_padding:
            action_is_pad = batch.get("action_is_pad")
            if action_is_pad is None:
                raise ValueError(
                    "do_mask_loss_for_padding=True requires 'action_is_pad' in the batch."
                )
            valid = ~action_is_pad                    # (B, horizon) True = real frame
            recon_mask = valid[:, 1:H]                # (B, horizon-1) for rolled ticks 0..horizon-2
            posterior_valid = valid                   # (B, horizon)
        else:
            recon_mask = None
            posterior_valid = torch.ones(B, H, dtype=torch.bool, device=state_full.device)

        # Posterior encodes the CLEAN demo (state, action) chunk; the train-time perturbation now
        # enters only at the rollout START (see `_pd_rollout(perturb_init=...)`), not here.
        x_seq = state_full[:, n_obs - 1 :]  # (B, H, state_dim) — demo state at each chunk-tick

        h = self.encode_observations(batch)                  # (B, cond_dim)

        if self.use_latent_z:
            # Posterior over the chunk: clean demo (state, action) pairs → q(z | demo trajectory).
            traj = torch.cat([x_seq, action_target], dim=-1)
            valid_mask = posterior_valid  # (B, horizon); ~action_is_pad when do_mask_loss_for_padding
            # pass h to the posterior only when z_uses_h.
            posterior_args = (traj, valid_mask, h) if self.z_uses_h else (traj, valid_mask)

            if self.use_vq:
                # 2-stage VQ-VAE spirit: prior is a pure observer of h — don't let its CE shape the encoder.
                prior_logits = self.prior(h.detach())
                z_e = self.posterior(*posterior_args)
                z_q_quant, idx_q = self.vq(z_e.unsqueeze(1))
                z_q = z_q_quant.squeeze(1)
                vq_indices = idx_q.squeeze(1).long()  # FSQ emits int32; CE/bincount want long
                # FSQ has no commitment loss (STE through the fixed grid) — only prior-CE is trained.
                # k detached on the CE: prior doesn't backprop into the posterior (van den Oord §3.2).
                vq_prior_ce_per_sample = F.cross_entropy(prior_logits, vq_indices.detach(), reduction="none")  # (B,)
                mu_p = sigma_p = mu_q = sigma_q = None
            else:
                mu_p, sigma_p = self.prior(h)
                mu_q, sigma_q = self.posterior(*posterior_args)
                eps_z = torch.randn(mu_q.shape, dtype=mu_q.dtype, device=mu_q.device)
                z_q = mu_q + sigma_q * eps_z
        else:
            mu_p = sigma_p = mu_q = sigma_q = None
            z_q = None

        # ---- Reconstruction: free-running rollout through the env's exact agent dynamics ----------
        # Integrate the drift's OWN predictions and match the rolled state path to the demo state path,
        # so a per-chunk z must bend the whole path onto the demo (identifiability fix for collapse).
        # perturb_init: train-only DART-style perturbation of the rollout START (targets stay clean).
        recon_per_sample = self._pd_rollout(
            state_full, h, z_q, perturb_init=self.training, recon_mask=recon_mask
        )  # (B,)

        recon_loss = recon_per_sample.mean()

        if self.use_latent_z:
            if self.use_vq:
                vq_prior_ce_loss = vq_prior_ce_per_sample.mean()
                other_loss = self.config.fsq_prior_weight * vq_prior_ce_loss
            else:
                kl_per_sample = _gaussian_kl_loss(
                    mu_q, sigma_q, mu_p, sigma_p, self.config.kl_min
                )  # (B,)
                kl_loss = kl_per_sample.mean()
                # ELBO-exact under the Gaussian state-observation model N(x̂(z), σ_obs²): no dt factor,
                # so kl_weight = 2·σ_obs² with σ_obs = trajectory observation std. Applied per-sample.
                kl_loss_weight = self.config.kl_weight / (H * action_dim)
                other_loss = (kl_loss_weight * kl_per_sample).mean()
            loss = recon_loss + other_loss
        else:
            loss = recon_loss
        # Rescale to velocity-error units
        dt = self.config.dt if self.config.dt is not None else 1.0
        loss_scale = 1.0 / dt**2
        loss = loss * loss_scale
        with torch.no_grad():
            loss_dict: dict[str, float] = {
                "recon_loss": recon_loss.detach().item() * loss_scale,
            }
            if self.use_latent_z:
                loss_dict["other_loss"] = other_loss.detach().item() * loss_scale
                # z_usage_gap: extra rollout recon from a batch-rolled (mismatched) z vs the true z,
                # both on CLEAN-init rollouts so the DART start-perturbation doesn't contaminate the gap.
                # ~0 ⇒ z ignored; measured over the whole rolled trajectory.
                base = self._pd_rollout(
                    state_full, h, z_q, perturb_init=False, recon_mask=recon_mask
                ).mean()
                z_rolled = torch.roll(z_q, shifts=1, dims=0)
                recon_loss_rolled = self._pd_rollout(
                    state_full, h, z_rolled, perturb_init=False, recon_mask=recon_mask
                ).mean()
                loss_dict["z_usage_gap"] = (recon_loss_rolled - base).item() * loss_scale
                if self.use_vq:
                    loss_dict["vq_prior_ce_loss"] = vq_prior_ce_loss.detach().item()
                    # Posterior code usage over the FSQ index space (num_codes = prod(levels)).
                    # Perplexity and active_codes are both capped by min(B, num_codes) within one
                    # batch — read them as a per-batch lower bound on utilization, not a fraction.
                    counts = torch.bincount(vq_indices, minlength=self.num_codes).float()
                    probs = counts / counts.sum().clamp_min(1.0)
                    entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum()
                    loss_dict["vq_perplexity"] = entropy.exp().item()
                    loss_dict["vq_active_codes"] = float((counts > 0).sum().item())
                    # Per-dim FSQ level usage — sweep-robust: each dim has ≤ max(levels) states
                    # (≪ batch), so unlike the flat index perplexity these are NOT batch-capped and
                    # are comparable across fsq_levels of different length/levels. Reported as means
                    # over dims of fractions in (0, 1]: usage = active_levels/level, perplexity =
                    # exp(H)/level (1/level ⇒ collapsed to one level, 1 ⇒ uniform over that dim).
                    level_idx = self.vq.indices_to_level_indices(vq_indices).long()  # (B, d), col j in [0, levels[j])
                    usage_fracs, ppl_fracs = [], []
                    for j, lvl in enumerate(self.config.fsq_levels):
                        cj = torch.bincount(level_idx[:, j], minlength=lvl).float()
                        pj = cj / cj.sum().clamp_min(1.0)
                        ppl_j = (-(pj * pj.clamp_min(1e-12).log()).sum()).exp()  # in [1, lvl]
                        usage_fracs.append((cj > 0).float().mean())              # active_levels / lvl
                        ppl_fracs.append(ppl_j / lvl)
                    loss_dict["fsq_level_usage"] = torch.stack(usage_fracs).mean().item()
                    loss_dict["fsq_level_perplexity"] = torch.stack(ppl_fracs).mean().item()
                    # Prior-side diversity: inference samples z ~ p(k|h), so a collapsed
                    # categorical prior is invisible in the posterior histogram above.
                    prior_marginal = F.softmax(prior_logits, dim=-1).mean(dim=0)
                    prior_entropy = -(prior_marginal * prior_marginal.clamp_min(1e-12).log()).sum()
                    loss_dict["vq_prior_perplexity"] = prior_entropy.exp().item()
                    prior_counts = torch.bincount(
                        prior_logits.argmax(dim=-1), minlength=self.num_codes
                    )
                    loss_dict["vq_prior_active_codes"] = float((prior_counts > 0).sum().item())
                else:
                    loss_dict["kl_loss"] = kl_loss.detach().item()
                    loss_dict["z_sigma_q_mean"] = sigma_q.mean().item()
                    loss_dict["z_sigma_p_mean"] = sigma_p.mean().item()
        return loss, loss_dict


# Per-episode latent z — prior p(z|h) and amortized posterior q(z|h, x_seq, a_seq).
# CVAE-style: train z ~ q via reparam, KL[q||p] regularizes the prior. At deployment z is
# resampled from p(z|h) in lock-step with every h refresh, committing each chunk to one mode.
# z conditions the drift net via FiLM alongside h (cond = concat([h, z])); the net input is x_aug only.

class StandardNormalPrior(nn.Module):
    """p(z) = N(0, I) — unconditional, parameter-free. Returns μ_p=0, log σ_p=0 for every h.

    Selected by `config.conditional_prior=False`. Recovers the standard VAE prior; with this
    choice z carries no h-dependent mode signal at deployment — each h-refresh samples a fresh
    mode index from N(0, I) regardless of the current image features. Useful as an ablation
    against the conditional `LatentPrior` (p(z|h))."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        zeros = torch.zeros(h.shape[0], self.z_dim, dtype=h.dtype, device=h.device)
        return zeros, torch.ones_like(zeros)  # (μ_p, σ_p) = (0, 1)


class LatentPrior(nn.Module):
    """p(z | h) — 2-layer MLP producing (μ_p, σ_p) from the image-only conditioning h."""

    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        sigma_activation: str,
        sigma_min: float,
    ):
        super().__init__()
        self.sigma_act = _sigma_act(sigma_activation)
        self.sigma_min = sigma_min
        self.trunk = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.sigma_head = nn.Linear(hidden_dim, z_dim)
        # Wide prior at init (σ_p ≈ 1 for exp, ≈ 0.69 for softplus) so KL doesn't over-constrain q.
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.trunk(h)
        mu = self.mu_head(feat)
        sigma = self.sigma_act(self.sigma_head(feat)) + self.sigma_min
        return mu, sigma


class LearnableCategoricalPrior(nn.Module):
    """Unconditional learnable prior over codebook indices: trainable logits θ ∈ ℝ^K.

    Trained via CE against the posterior's (detached) index; softmax(θ) converges to the
    empirical marginal p(k). Init zeros → uniform at start.
    """

    def __init__(self, codebook_size: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(codebook_size))

    def forward(self, h: Tensor) -> Tensor:
        return self.logits.unsqueeze(0).expand(h.shape[0], -1)


class LatentPriorVQ(nn.Module):
    """p(k | h) over codebook indices, parameterised as a 2-layer MLP classifier.

    Trained via CE against the posterior's (detached) index. Head init zeros → uniform prior at init.
    """

    def __init__(self, h_dim: int, codebook_size: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.head = nn.Linear(hidden_dim, codebook_size)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: Tensor) -> Tensor:
        return self.head(self.trunk(h))


# Trajectory-encoder posterior (per_chunk): a dilated TCN (Bai et al. 2018) over
# concat([state, action]) — 1×1 lift + kernel-3 dilated residual blocks + masked mean-pool. Pads
# are zeroed before every conv and GroupNorm is masked, so eval outputs are bit-equivalent to
# exact-length (no cross-pad leak). RF = 1 + 4·(2^num_levels − 1); num_levels is auto-sized from the
# chunk length (horizon) in LatentODEModel.__init__ so RF ≈ horizon.
class _TCNResidualBlock(nn.Module):
    """(DilatedConv3 → masked GroupNorm → Mish) × 2 + identity skip; centered padding."""

    def __init__(self, channels: int, dilation: int, n_groups: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = _MaskedGroupNorm(n_groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = _MaskedGroupNorm(n_groups, channels)
        self.act = nn.Mish()

    def forward(self, h: Tensor, valid_mask: Tensor, m: Tensor) -> Tensor:
        # Zero pads before each conv so the dilated kernel never reads leaked pad values.
        out = self.act(self.norm1(self.conv1(h * m), valid_mask)) * m
        out = self.act(self.norm2(self.conv2(out), valid_mask)) * m
        return out + h


class _TrajEncoder(nn.Module):
    """Dilated-TCN encoder + masked mean-pool → (B, hidden_dim). Kernel 3; depth = num_levels."""

    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int, n_groups: int = 8):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)  # pointwise channel lift
        self.blocks = nn.ModuleList(
            [_TCNResidualBlock(hidden_dim, dilation=2 ** level, n_groups=n_groups) for level in range(num_levels)]
        )
        self.post_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())

    def forward(self, traj: Tensor, valid_mask: Tensor) -> Tensor:
        # traj: (B, T, state_dim + action_dim); valid_mask: (B, T) bool.
        m = valid_mask.to(traj.dtype).unsqueeze(1)
        h = self.input_proj(traj.transpose(1, 2) * m)
        for block in self.blocks:
            h = block(h, valid_mask, m)
        return self.post_pool(_masked_mean_pool(h, valid_mask))


class LatentPosteriorTraj(nn.Module):
    """q(z | x_{0:T}, a_{0:T}, [h]) — (state, action)-traj encoder pooled feats (concat h iff h_dim>0) → (μ, σ).

    Concats the chunk h when h_dim>0 (z_uses_h); h_dim=0 → trajectory-only.
    """

    def __init__(
        self,
        input_dim: int,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        sigma_activation: str,
        sigma_min: float,
        num_levels: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.sigma_act = _sigma_act(sigma_activation)
        self.sigma_min = sigma_min
        self.encoder = _TrajEncoder(input_dim, hidden_dim, num_levels, n_groups)
        head_in = hidden_dim + h_dim
        self.mu_head = nn.Linear(head_in, z_dim)
        self.sigma_head = nn.Linear(head_in, z_dim)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

    def forward(self, traj: Tensor, valid_mask: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        feat = self.encoder(traj, valid_mask)
        if h is not None:
            feat = torch.cat([feat, h], dim=-1)
        mu = self.mu_head(feat)
        sigma = self.sigma_act(self.sigma_head(feat)) + self.sigma_min
        return mu, sigma


class LatentPosteriorTrajVQ(nn.Module):
    """Deterministic VQ posterior — (state, action)-traj encoder pooled feats (concat h iff h_dim>0) → z_e."""

    def __init__(
        self,
        input_dim: int,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        num_levels: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.encoder = _TrajEncoder(input_dim, hidden_dim, num_levels, n_groups)
        self.head = nn.Linear(hidden_dim + h_dim, z_dim)

    def forward(self, traj: Tensor, valid_mask: Tensor, h: Tensor | None = None) -> Tensor:
        feat = self.encoder(traj, valid_mask)
        if h is not None:
            feat = torch.cat([feat, h], dim=-1)
        return self.head(feat)


def _masked_mean_pool(h: Tensor, valid_mask: Tensor) -> Tensor:
    """Mean of `h` (B, C, T) over T positions where `valid_mask` (B, T) is True."""
    mask_f = valid_mask.to(h.dtype).unsqueeze(1)
    summed = (h * mask_f).sum(dim=-1)
    counts = mask_f.sum(dim=-1).clamp_min(1.0)
    return summed / counts


class _MaskedGroupNorm(nn.Module):
    """GroupNorm with per-(batch, group) mean/var over `valid_mask` positions only.

    Padded positions still receive the affine transform; caller must zero pads before the next
    Conv1d so leaked values from kernel-3 don't pollute valid outputs.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels={num_channels} not divisible by num_groups={num_groups}.")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        # x: (B, C, T); valid_mask: (B, T) bool
        B, C, T = x.shape
        G = self.num_groups
        Cg = C // G
        m = valid_mask.to(x.dtype).view(B, 1, 1, T)
        x_g = x.view(B, G, Cg, T)
        n_valid = m.sum(dim=-1, keepdim=True).clamp_min(1.0)    # guard all-pad samples
        count = Cg * n_valid
        mean = (x_g * m).sum(dim=(2, 3), keepdim=True) / count
        diff = (x_g - mean) * m
        var = (diff * diff).sum(dim=(2, 3), keepdim=True) / count
        x_norm = (x_g - mean) / (var + self.eps).sqrt()
        x_norm = x_norm.view(B, C, T)
        return x_norm * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)


def _gaussian_kl_loss(
    mu_q: Tensor, sigma_q: Tensor, mu_p: Tensor, sigma_p: Tensor, kl_min: float = 0.0
) -> Tensor:
    """KL[N(μ_q, diag σ_q²) || N(μ_p, diag σ_p²)] per sample, summed over latent dim.

    kl_min: Per-dim KL floor in nats (free bits). Each dim's KL is clamped to
        max(kl, kl_min) before summing, so the model uses up to kl_min nats
        per dim without KL penalty, preventing posterior collapse.

        For a task with N modes, set kl_min ≈ log(N) / z_dim so the total
        budget z_dim × kl_min ≈ log(N) nats covers the mode bits.
        E.g., N=2 → 0.35 (z_dim=2), 0.04 (z_dim=16). Default 0 disables.
    """
    var_q = sigma_q.pow(2)
    var_p = sigma_p.pow(2)
    per_dim = sigma_p.log() - sigma_q.log() + 0.5 * (var_q + (mu_q - mu_p) ** 2) / var_p - 0.5
    if kl_min > 0.0:
        per_dim = per_dim.clamp_min(kl_min)
    return per_dim.sum(dim=-1)


class LatentODEDriftNet(nn.Module):
    """Point-wise hourglass MLP with per-block FiLM conditioning — port of
    DiffusionConditionalUnet1d (horizon-axis Conv1d → Linear).

    Inputs:
        x:    (B, input_dim)     — augmented state (n_obs_steps frames flattened);
                                   the net reads local first-differences ≈ velocity.
        cond: (B, cond_dim)      — global conditioning concat([h, z]) (FiLM); z omitted when no latent.
        step: (B,) | None        — step-distance from h; sinusoidally embedded onto the cond when use_pe.
    Output:
        mu:   (B, action_dim) — drift. Action σ is not learned; inference is deterministic (x_d = x + μ·dt).

    Width ladder mirrors DiffusionConditionalUnet1d's down_dims hourglass; the "mid" block
    keeps width at d_{L-1} (matches the two mid_modules in the U-Net).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        cond_dim: int,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
        use_pe: bool = False,
        pe_dim: int = 64,
    ):
        super().__init__()
        assert len(down_dims) >= 1, "`down_dims` must contain at least one width."

        # Step-distance encoder (DP's diffusion_step_encoder): sinusoidal embed → MLP, concatenated
        # onto the FiLM cond. Widens cond_dim by pe_dim.
        self.use_pe = use_pe
        if use_pe:
            self.step_encoder = nn.Sequential(
                DiffusionSinusoidalPosEmb(pe_dim),
                nn.Linear(pe_dim, pe_dim * 4),
                nn.Mish(),
                nn.Linear(pe_dim * 4, pe_dim),
            )
            cond_dim = cond_dim + pe_dim

        widths = [input_dim, *down_dims, down_dims[-1], *reversed(down_dims[:-1])]
        self.blocks = nn.ModuleList(
            [
                FiLMResidualMLPBlock(
                    in_dim=widths[i],
                    out_dim=widths[i + 1],
                    cond_dim=cond_dim,
                    n_groups=n_groups,
                    use_film_scale_modulation=use_film_scale_modulation,
                )
                for i in range(len(widths) - 1)
            ]
        )

        self.final_norm = nn.GroupNorm(n_groups, widths[-1])
        self.final_act = nn.Mish()
        self.mu_head = nn.Linear(widths[-1], action_dim)

    def forward(self, x: Tensor, cond: Tensor, step: Tensor | None = None) -> Tensor:
        if self.use_pe:
            cond = torch.cat([self.step_encoder(step), cond], dim=-1)  # cat([step_embed, h]), as in DP
        feat = x
        for block in self.blocks:
            feat = block(feat, cond)
        feat = self.final_act(self.final_norm(feat.unsqueeze(-1)).squeeze(-1))
        return self.mu_head(feat)


class FiLMResidualMLPBlock(nn.Module):
    """Point-wise ResNet block with FiLM-with-scale conditioning.

    Mirrors DiffusionConditionalResidualBlock1d (Conv1d → Linear since the ODE acts on a
    single time step):
        x ──► Linear ──► GroupNorm ──► Mish ──► (* scale + bias from cond) ──►
              Linear ──► GroupNorm ──► Mish ──► (+ residual)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cond_dim: int,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_dim = out_dim

        self.lin1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.GroupNorm(n_groups, out_dim)
        self.act1 = nn.Mish()

        cond_channels = out_dim * 2 if use_film_scale_modulation else out_dim
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.lin2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.GroupNorm(n_groups, out_dim)
        self.act2 = nn.Mish()

        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    @staticmethod
    def _groupnorm_pointwise(norm: nn.GroupNorm, feat: Tensor) -> Tensor:
        # nn.GroupNorm expects (B, C, *spatial); treat the vector as (B, C, 1).
        return norm(feat.unsqueeze(-1)).squeeze(-1)

    def forward(self, feat: Tensor, cond: Tensor) -> Tensor:
        out = self.lin1(feat)
        out = self._groupnorm_pointwise(self.norm1, out)
        out = self.act1(out)

        cond_embed = self.cond_encoder(cond)
        if self.use_film_scale_modulation:
            scale, bias = cond_embed.chunk(2, dim=-1)
            out = scale * out + bias
        else:
            out = out + cond_embed

        out = self.lin2(out)
        out = self._groupnorm_pointwise(self.norm2, out)
        out = self.act2(out)

        return out + self.residual_proj(feat)
