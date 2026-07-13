#!/usr/bin/env python
#
# LatentSDE Policy — point-wise drift / diffusion network.
#
# Port of DiffusionConditionalUnet1d to the SDE setting: horizon-axis Conv1d → Linear,
# diffusion-timestep encoder removed (no denoising loop), U-Net skip-connections collapse
# into per-block residuals. FiLM-with-scale / GroupNorm / Mish / down_dims widths / FiLM
# conditioning are preserved verbatim for architectural fairness vs. DiffusionPolicy.

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..diffusion.modeling_diffusion import DiffusionRgbEncoder, DiffusionSinusoidalPosEmb
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_latent_sde import LatentSDEConfig


def _sigma_act(activation: str):
    """σ = act(s) + sigma_min; caller adds the floor. Used by z prior/posterior heads."""
    if activation == "exp":
        return torch.exp
    if activation == "softplus":
        return F.softplus
    raise ValueError(f"sigma_activation must be 'exp' or 'softplus'; got {activation!r}.")


class LatentSDEPolicy(PreTrainedPolicy):
    """LatentSDE policy (free-space, no compliance) — first PoC.

    Wraps `LatentSDEModel` and implements the LeRobot policy interface
    (reset / select_action / forward) with DiffusionPolicy-style observation queues.

    Inference duty cycle (research_brief.md §1.2): h (image conditioning) and z
    (per-episode latent) are refreshed together every `n_action_steps` ticks —
    matching DP's vision-encoder cadence. The drift/diffusion net runs every tick
    on the current measured state.
    """

    config_class = LatentSDEConfig
    name = "latent_sde"

    def __init__(self, config: LatentSDEConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self._cached_h: Tensor | None = None
        self._steps_until_h_refresh: int = 0
        self._cached_z: Tensor | None = None

        self.model = LatentSDEModel(config)
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
        # Single-step SDE policy has no chunk-at-once inference path; rollout is per-tick.
        raise NotImplementedError(
            "LatentSDEPolicy does not support predict_action_chunk (chunked / async inference); "
            "use select_action for per-tick rollout."
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """One SDE step per tick with per-tick state-feedback.

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
        action = self.model.step(x_aug, self._cached_h, self._cached_z, noise=noise, step_idx=step_idx)
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


class LatentSDEModel(nn.Module):
    """Assembles h, holds drift/diffusion net, exposes sample + compute_loss.

    Mirrors the role of DiffusionModel one-for-one.
    """

    def __init__(self, config: LatentSDEConfig):
        super().__init__()
        self.config = config

        # h is image-only by design: proprio state lives on the fast clock as the SDE
        # input x (the n_obs window is flattened in `select_action`); env_state is deferred.
        if not self.config.image_features:
            raise ValueError(
                "LatentSDEPolicy requires at least one image feature (h = image-only "
                "in this PoC). Got input_features with no observation.image*."
            )

        global_cond_dim = 0
        num_images = len(self.config.image_features)
        if self.config.use_separate_rgb_encoder_per_camera:
            # DiffusionRgbEncoder reads a few fields from the config; LatentSDEConfig matches
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

            # TCN posterior depth auto-sized so its receptive field (RF ≈ 4·2^L) matches the chunk
            # length (horizon), kernel 3. E.g. horizon 8→1, 16→2, 32→3, 64→4.
            tcn_levels = max(1, round(math.log2(config.horizon / 4)))

            if self.use_vq:
                # Discrete latent (config.quantizer): FSQ or VQ. num_codes = size of the flat index
                # space the categorical prior/CE/perplexity operate over.
                if config.quantizer == "fsq":
                    self.num_codes = math.prod(config.fsq_levels)  # prod(levels)
                else:  # "vq"
                    self.num_codes = config.vq_codebook_size
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
                if config.quantizer == "fsq":
                    from vector_quantize_pytorch import FSQ
                    # FSQ: bounded scalar grid + straight-through rounding. No learnable codebook,
                    # no commitment loss, no dead codes — z_dim is fixed to len(levels) (config).
                    self.vq = FSQ(levels=config.fsq_levels)
                else:  # "vq"
                    from vector_quantize_pytorch import VectorQuantize
                    self.vq = VectorQuantize(
                        dim=self.z_dim,
                        codebook_size=config.vq_codebook_size,
                        decay=config.vq_decay,
                        commitment_weight=config.vq_commit_weight,
                        use_cosine_sim=True,  # STE: cosine distance is more stable than L2 for high-dim z
                        rotation_trick=False,  # STE: forward z_q == raw code, so train matches inference (get_output_from_indices)
                        kmeans_init=True,  # seed codes from data, not random Gaussian (avoids born-dead codes)
                        threshold_ema_dead_code=2,  # revive codes whose EMA usage dies, countering codebook collapse
                    )
                    # NOTE: kmeans_init seeds K centroids from ONE batch of z_e (B vectors). If
                    # vq_codebook_size > batch_size, the surplus codes start unseeded — keep K <= batch_size.
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

        # z routing (z_mode): "cond" → z widens the FiLM cond ([h, z], net input = x_aug, legacy);
        # "input" → z concatenated onto the net input ([x_aug, z], cond = h). No-op when z_dim == 0.
        # Outputs drift μ only — action σ is not learned; inference noise uses √(kl_weight/2).
        self.z_as_input = self.use_latent_z and config.z_mode == "input"
        self.net = LatentSDEDriftDiffusionNet(
            input_dim=self.state_dim * config.n_obs_steps + (self.z_dim if self.z_as_input else 0),
            action_dim=config.action_feature.shape[0],
            cond_dim=self.h_dim + (0 if self.z_as_input else self.z_dim),
            down_dims=config.down_dims,
            n_groups=config.n_groups,
            use_film_scale_modulation=config.use_film_scale_modulation,
            use_pe=config.use_pe,
            pe_dim=config.pe_dim,
        )

        if config.compile_model:
            self.net = torch.compile(self.net, mode=config.compile_mode)

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

    def _effective_sigma(self) -> float:
        """SDE diffusion coefficient σ_eff = √(kl_weight / 2). With the ELBO-exact KL scaling
        in compute_loss (`/(H·D·dt)`), kl_weight = 2·σ_eff² holds exactly. Position noise per
        SDE step = σ_eff·√dt."""
        return math.sqrt(self.config.kl_weight / 2.0)

    def _sde_step(
        self,
        x_now: Tensor,
        mu: Tensor,
        dt: float,
        deterministic: bool,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        mean = x_now + mu * dt
        if deterministic:
            return mean
        std = self._effective_sigma() * (dt ** 0.5)
        if noise is not None:
            if noise.shape != mean.shape:
                raise ValueError(
                    f"`noise` must have shape {tuple(mean.shape)} (matching mean); got {tuple(noise.shape)}."
                )
            eps = noise.to(dtype=mean.dtype, device=mean.device)
        else:
            eps = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + std * eps

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
            # FSQ maps flat index → grid code; VQ looks the code up in its learnable codebook.
            if self.config.quantizer == "fsq":
                return self.vq.indices_to_codes(k)
            return self.vq.get_output_from_indices(k)
        else:
            mu_p, sigma_p = self.prior(h)
            if deterministic:
                return mu_p
            eps = torch.randn(mu_p.shape, dtype=mu_p.dtype, device=mu_p.device, generator=generator)
            return mu_p + sigma_p * eps

    def _drift_inputs(self, x_aug: Tensor, h: Tensor, z: Tensor | None) -> tuple[Tensor, Tensor]:
        """Assemble the drift net's (input, FiLM cond) per z routing (config.z_mode).

        "cond" (default): input = x_aug, cond = [h, z]. "input": input = [x_aug, z], cond = h.
        z=None (no latent) → (x_aug, h).
        """
        if not self.use_latent_z or z is None:
            return x_aug, h
        if self.z_as_input:
            return torch.cat([x_aug, z], dim=-1), h
        return x_aug, torch.cat([h, z], dim=-1)

    def step(
        self,
        x_aug: Tensor,
        h: Tensor,
        z: Tensor | None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        step_idx: int = 0,
    ) -> Tensor:
        """One Euler-Maruyama step from pre-computed h (and optional z).

        Args:
            x_aug: (B, n_obs_steps * state_dim) — augmented state input. The drift/diffusion
                   net reads the full window; the residual update is anchored to the most
                   recent frame, sliced as `x_aug[..., -state_dim:]`.
            noise: (B, action_dim) optional pre-sampled Brownian increment. Mirrors
                   `DiffusionModel.conditional_sample(noise=...)`: when given, replaces the
                   internal `randn`; ignored when `deterministic_inference` is True.
            step_idx: ticks since the current h refresh (0..n_action_steps-1); the PE index when use_pe.
        """
        net_in, cond = self._drift_inputs(x_aug, h, z)
        step = (
            torch.full((x_aug.shape[0],), float(step_idx), dtype=x_aug.dtype, device=x_aug.device)
            if self.config.use_pe
            else None
        )
        mu = self.net(net_in, cond, step)
        dt = self.config.sde_dt if self.config.sde_dt is not None else 1.0
        x_now = x_aug[..., -self.state_dim:]
        return self._sde_step(
            x_now,
            mu,
            dt=dt,
            deterministic=self.config.deterministic_inference,
            generator=generator,
            noise=noise,
        )

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """β-VAE-style loss: recon = mean‖μ − v*‖² + (kl_weight / (H·D·dt)) · KL[q‖p],
        with v* = (a − x)/dt the empirical one-step velocity.

        Recon is the per-element velocity MSE. The KL scale is ELBO-exact under the SDE
        decoder Δx ~ N(μ·dt, σ²·dt), giving kl_weight = 2·σ_eff² where σ_eff = √(kl_weight/2)
        is the SDE diffusion coefficient (also used at inference: position noise = σ_eff·√dt).

        x_seq is the measured state trajectory from the dataset (no teacher-forcing). Train and
        inference see the same state distribution only when state_noise_std==0; under state-noise the
        drift input and the posterior input are perturbed at train time (corrective/consistency aug),
        while inference stays clean.

        Expected `batch` (normalized + on device; LatentSDEPolicy.forward stacks images):
            "observation.state":  (B, n_obs_steps + horizon - 1, state_dim) — past + future
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
            "action":             (B, horizon, action_dim)
            "action_is_pad":      (B, horizon) — used iff do_mask_loss_for_padding

        Padding handling: `drop_n_last_frames` keeps the EXECUTED region unpadded, but the
        predicted tail may be copy-padded at episode ends. When do_mask_loss_for_padding=True,
        those ticks are masked out of the recon MSE and the posterior valid_mask
        (`valid = ~action_is_pad`); otherwise they are included unmasked (DP-style default).
        """
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch

        action_target = batch[ACTION]                        # (B, H, action_dim)
        B, H, action_dim = action_target.shape
        n_obs = self.config.n_obs_steps
        assert self.config.horizon == H, (
            f"compute_loss expects H == horizon action targets; got H={H} "
            f"vs config.horizon={self.config.horizon}."
        )

        # state_full: (B, n_obs + H - 1, state_dim). x_seq[k] = state at chunk-tick k;
        # x_seq_aug[k] = flatten(x[k - n_obs + 1], ..., x[k]) — the SDE drift input.
        state_full = batch[OBS_STATE]
        expected_T = n_obs + H - 1
        if state_full.shape[1] != expected_T:
            raise ValueError(
                f"Expected batch[OBS_STATE] T={expected_T} (n_obs_steps + H - 1); "
                f"got T={state_full.shape[1]}. Check observation_delta_indices_per_key."
            )
        dt = self.config.sde_dt if self.config.sde_dt is not None else 1.0
        noise = self.training and self.config.state_noise_std > 0
        std = self.config.state_noise_std * dt**0.5

        # Drift input (train-only state-noise): perturb the state window, then recompute the recon
        # target from the perturbed anchor (corrective drift). The posterior sees the same noised
        # state (below). state_noise_schedule: "uniform" = std·√dt on every frame; "linear" = std
        # ramps std·√dt/H → std·√dt across chunk ticks 0..H-1 (indices 1..H; tick 0 gets std·√dt/H,
        # NOT zero; peak at t=H-1), past obs frames (delta<0) get 0.
        if not noise:
            state_win = state_full
        elif self.config.state_noise_schedule == "linear":
            delta = torch.arange(expected_T, device=state_full.device, dtype=state_full.dtype) - (n_obs - 1)
            std_t = std * (delta + 1).clamp(min=0) / H   # (T,) std/H at tick 0 → std at tick H-1 (idx 1..H)
            state_win = state_full + std_t.view(1, -1, 1) * torch.randn_like(state_full)
        else:  # "uniform"
            state_win = state_full + std * torch.randn_like(state_full)

        x_seq = state_win[:, n_obs - 1 :]  # (B, H, state_dim) — possibly noised anchor (drift input)
        x_seq_clean = state_full[:, n_obs - 1 :]  # (B, H, state_dim) — clean demo anchor x_k
        x_seq_aug = state_win.unfold(1, n_obs, 1).permute(0, 1, 3, 2).reshape(B, H, n_obs * self.state_dim)

        h = self.encode_observations(batch)                  # (B, cond_dim)

        # Padding mask (DP-style). `action_is_pad` (B, H) marks copy-padded chunk ticks at episode
        # ends; it aligns with the action/state deltas 0..H-1. Tick k's recon uses action_target[:, k]
        # and x_seq[:, k] (both delta k), and the posterior entry at tick k is delta k too, so the
        # per-tick mask is valid[:, k]. Off → all-valid (identical to the legacy no-mask behavior).
        if self.config.do_mask_loss_for_padding:
            action_is_pad = batch.get("action_is_pad")
            if action_is_pad is None:
                raise ValueError(
                    "do_mask_loss_for_padding=True requires 'action_is_pad' in the batch."
                )
            valid = ~action_is_pad                           # (B, H) True = real frame
        else:
            valid = torch.ones(B, H, dtype=torch.bool, device=x_seq.device)

        if self.use_latent_z:
            # Posterior over the chunk: posterior_state selects the clean or the noised demo state (the
            # action is always the clean demo action). "clean" ⇒ same demo → same z regardless of the
            # train-time state-noise; "noisy" ⇒ legacy (the same noised state the drift reads).
            post_state = x_seq_clean if self.config.posterior_state == "clean" else x_seq
            traj = torch.cat([post_state, action_target], dim=-1)
            valid_mask = valid                               # (B, H); ~action_is_pad when masking
            # pass h to the posterior only when z_uses_h.
            posterior_args = (traj, valid_mask, h) if self.z_uses_h else (traj, valid_mask)

            if self.use_vq:
                # 2-stage VQ-VAE spirit: prior is a pure observer of h — don't let its CE shape the encoder.
                prior_logits = self.prior(h.detach())
                z_e = self.posterior(*posterior_args)
                if self.config.quantizer == "fsq":
                    # FSQ: STE through the fixed grid — no learnable codebook, no commitment loss.
                    z_q_quant, idx_q = self.vq(z_e.unsqueeze(1))
                    vq_commit_loss = None
                else:  # "vq": the lib returns commitment_weight·mse(z_e, sg[z_q]) directly (EMA codebook,
                    # no orthogonal/diversity/CE reg) — use it as-is. The old per-sample recompute existed
                    # only for the now-removed per_episode H/T_ep per-element weighting.
                    z_q_quant, idx_q, vq_commit_loss = self.vq(z_e.unsqueeze(1))
                z_q = z_q_quant.squeeze(1)
                vq_indices = idx_q.squeeze(1).long()  # FSQ emits int32; CE/bincount want long
                # k detached on the CE: prior doesn't backprop into the posterior/codebook (van den Oord §3.2).
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

        flat_x_aug = x_seq_aug.reshape(B * H, -1)
        flat_h = h.unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
        # Per-tick step-distance k=0..H-1 (row b*H+k ↦ k, matching the flatten above); None if use_pe off.
        flat_step = (
            torch.arange(H, dtype=h.dtype, device=h.device).unsqueeze(0).expand(B, H).reshape(B * H)
            if self.config.use_pe
            else None
        )
        flat_z = None
        if self.use_latent_z:
            flat_z = z_q.unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
        flat_net_in, flat_cond = self._drift_inputs(flat_x_aug, flat_h, flat_z)

        mu_flat = self.net(flat_net_in, flat_cond, flat_step)
        mu = mu_flat.reshape(B, H, action_dim)

        # Recon target v* = (a_anchor − x̃)/dt: one-step full return to a demo action from the noised
        # anchor x̃ = x_seq. action_anchor picks a_anchor:
        #   "clean"   — the corresponding-index action a_k (== legacy corrective target).
        #   "nearest" — the action a_j of the nearest demo state x_j over the chunk (autonomous field; z
        #               resolves the branch ambiguity at self-intersections). Equals "clean" with no tube.
        if self.config.action_anchor == "nearest":
            nn_idx = torch.cdist(x_seq, x_seq_clean).argmin(dim=-1)  # (B,H) nearest clean-state index
            a_anchor = torch.gather(action_target, 1, nn_idx.unsqueeze(-1).expand(-1, -1, action_dim))
        else:  # "clean"
            a_anchor = action_target
        target_velocity = (a_anchor - x_seq) / dt
        # Masked MSE: zero copy-padded ticks, then normalize by NOMINAL B·H·D (not the valid count as
        # DP does) — this keeps the global recon/KL constant uniform so kl_weight = 2·σ_eff² stays
        # ELBO-exact, and reduces to the plain mean when nothing is masked. `valid` is all-True unless
        # do_mask_loss_for_padding.
        recon_se = (mu - target_velocity) ** 2               # (B, H, action_dim)
        if self.config.do_mask_loss_for_padding:
            recon_se = recon_se * valid.unsqueeze(-1)
        recon_loss = recon_se.mean()

        if self.use_latent_z:
            if self.use_vq:
                vq_prior_ce_loss = vq_prior_ce_per_sample.mean()
                if self.config.quantizer == "vq":
                    # vq_commit_loss already carries commitment_weight (applied inside VectorQuantize).
                    other_loss = vq_commit_loss + self.config.vq_prior_weight * vq_prior_ce_loss
                else:  # "fsq" — no commitment loss, only prior-CE
                    other_loss = self.config.fsq_prior_weight * vq_prior_ce_loss
            else:
                kl_per_sample = _gaussian_kl_loss(
                    mu_q, sigma_q, mu_p, sigma_p, self.config.kl_min
                )  # (B,)
                kl_loss = kl_per_sample.mean()
                # Merged KL weight: ELBO-exact β/(H·D·dt), so kl_weight = 2·σ_eff². Applied per-sample.
                kl_loss_weight = self.config.kl_weight / (H * action_dim * dt)
                other_loss = (kl_loss_weight * kl_per_sample).mean()
            loss = recon_loss + other_loss
        else:
            loss = recon_loss

        with torch.no_grad():
            loss_dict: dict[str, float] = {
                "recon_loss": recon_loss.detach().item(),
                "effective_sigma": self._effective_sigma(),
            }
            if self.use_latent_z:
                loss_dict["other_loss"] = other_loss.detach().item()
                # z_usage_gap: extra recon error from a batch-rolled (mismatched) z. ~0 ⇒ z ignored.
                flat_z_rolled = (
                    torch.roll(z_q, shifts=1, dims=0).unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
                )
                net_in_rolled, cond_rolled = self._drift_inputs(flat_x_aug, flat_h, flat_z_rolled)
                mu_rolled = self.net(net_in_rolled, cond_rolled, flat_step)
                rolled_se = (mu_rolled.reshape(B, H, action_dim) - target_velocity) ** 2
                if self.config.do_mask_loss_for_padding:
                    rolled_se = rolled_se * valid.unsqueeze(-1)
                recon_loss_rolled = rolled_se.mean()
                loss_dict["z_usage_gap"] = (recon_loss_rolled - recon_loss).item()
                # recon_loss_prior: recon with z ~ p(z|h) — the DEPLOY-time z (training uses q). The
                # posterior's recon gain transfers to deployment only if this stays near recon_loss; if
                # it rises toward the rolled (mismatched-z) level, the gain is posterior LEAKAGE — z
                # encodes trajectory info the prior can't reproduce. prior_recon_gap = the deploy penalty.
                z_prior = self.sample_z_from_prior(h)  # (B, z_dim), deploy prior distribution
                flat_z_prior = z_prior.unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
                net_in_p, cond_p = self._drift_inputs(flat_x_aug, flat_h, flat_z_prior)
                mu_prior = self.net(net_in_p, cond_p, flat_step).reshape(B, H, action_dim)
                prior_se = (mu_prior - target_velocity) ** 2
                if self.config.do_mask_loss_for_padding:
                    prior_se = prior_se * valid.unsqueeze(-1)
                recon_loss_prior = prior_se.mean()
                loss_dict["recon_loss_prior"] = recon_loss_prior.item()
                loss_dict["prior_recon_gap"] = (recon_loss_prior - recon_loss).item()
                if self.use_vq:
                    loss_dict["vq_prior_ce_loss"] = vq_prior_ce_loss.detach().item()
                    if self.config.quantizer == "vq":
                        # Weighted commit term (= commitment_weight · mse) as it enters other_loss.
                        loss_dict["vq_commit_loss"] = vq_commit_loss.detach().item()
                    # Posterior code usage over the flat index space (num_codes = prod(fsq_levels) for
                    # FSQ, vq_codebook_size for VQ). Perplexity and active_codes are both capped by
                    # min(B, num_codes) within one batch — read them as a per-batch lower bound on
                    # utilization, not a fraction.
                    counts = torch.bincount(vq_indices, minlength=self.num_codes).float()
                    probs = counts / counts.sum().clamp_min(1.0)
                    entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum()
                    loss_dict["vq_perplexity"] = entropy.exp().item()
                    loss_dict["vq_active_codes"] = float((counts > 0).sum().item())
                    if self.config.quantizer == "fsq":
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
# exact-length (no cross-pad leak). RF = 1 + 4·(2^num_levels − 1).
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


class LatentSDEDriftDiffusionNet(nn.Module):
    """Point-wise hourglass MLP with per-block FiLM conditioning — port of
    DiffusionConditionalUnet1d (horizon-axis Conv1d → Linear).

    Inputs:
        x:    (B, input_dim)     — augmented state (n_obs_steps frames flattened);
                                   the net reads local first-differences ≈ velocity.
        cond: (B, cond_dim)      — global conditioning concat([h, z]) (FiLM); z omitted when no latent.
        step: (B,) | None        — step-distance from h; sinusoidally embedded onto the cond when use_pe.
    Output:
        mu:   (B, action_dim) — SDE drift. Action σ is not learned; inference noise uses
                                √(kl_weight/2) from the config.

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

    Mirrors DiffusionConditionalResidualBlock1d (Conv1d → Linear since the SDE acts on a
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
