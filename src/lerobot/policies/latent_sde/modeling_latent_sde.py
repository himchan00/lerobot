#!/usr/bin/env python
#
# LatentSDE Policy — point-wise drift / diffusion network.
#
# Port of DiffusionConditionalUnet1d to the SDE setting: horizon-axis Conv1d → Linear,
# diffusion-timestep encoder removed (no denoising loop), U-Net skip-connections collapse
# into per-block residuals. FiLM-with-scale / GroupNorm / Mish / down_dims widths / FiLM
# conditioning are preserved verbatim for architectural fairness vs. DiffusionPolicy.

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..diffusion.modeling_diffusion import DiffusionRgbEncoder
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_latent_sde import LatentSDEConfig


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

    def set_train_dataset(self, dataset) -> None:
        """Forward to the model (trainer calls this once after dataset construction)."""
        self.model.set_train_dataset(dataset)

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
        """Stateless 1-step action wrapped as a length-1 "chunk" for API parity.

        Returns shape (B, 1, action_dim). Bypasses the h/z cache; use `select_action`
        for interactive rollout.
        """
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        action = self.model.generate_action(batch, noise=noise)
        return action.unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """One SDE step per tick with per-tick state-feedback.

        Refresh h every `n_action_steps` ticks. per_chunk re-samples z alongside h;
        per_episode samples z once per reset() and holds it for the whole rollout.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        # Slow path: re-encode h every n_action_steps ticks. per_chunk re-samples z with h;
        # per_episode samples once (held until reset() clears _cached_z).
        if self._cached_h is None or self._steps_until_h_refresh == 0:
            stacked_images = torch.stack(list(self._queues[OBS_IMAGES]), dim=1)
            self._cached_h = self.model.encode_observations({OBS_IMAGES: stacked_images})
            if not self.model.per_episode_z or self._cached_z is None:
                self._cached_z = self.model.sample_z_from_prior(self._cached_h)
            self._steps_until_h_refresh = self.config.n_action_steps

        # Fast path: flatten the n_obs_steps state window into x_aug so the drift can read
        # local first-differences ≈ velocity (mirrors DP's state branch in the global cond).
        stacked_state = torch.stack(list(self._queues[OBS_STATE]), dim=1)  # (B, n_obs, D)
        x_aug = stacked_state.reshape(stacked_state.shape[0], -1)           # (B, n_obs * D)
        action = self.model.step(x_aug, self._cached_h, self._cached_z, noise=noise)
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
        # input x (see `_augmented_state`); env_state is deferred (see LatentSDEConfig docstring).
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

        # use_latent_z=False → no prior/posterior/vq, z_dim=0. Otherwise the three modules below
        # branch on independent axes:
        #   * self.prior     : (use_vq) × (config.conditional_prior)
        #   * self.posterior : (use_vq) × (per_episode_z) — Traj* sees full episode, others H-chunk
        #   * self.vq        : built iff use_vq
        self.use_vq = config.use_vq
        self.per_episode_z = (config.z_sampling_mode == "per_episode")
        self.state_dim = config.robot_state_feature.shape[0]
        if not self.use_latent_z:
            self.z_dim = 0
            self.prior = None
            self.posterior = None
            self.vq = None
        else:
            self.z_dim = config.z_dim
            posterior_hidden = config.z_posterior_hidden_dim or self.h_dim
            prior_hidden = config.z_prior_hidden_dim or self.h_dim

            if self.use_vq:
                self.prior = (
                    LatentPriorVQ(
                        h_dim=self.h_dim,
                        codebook_size=config.vq_codebook_size,
                        hidden_dim=prior_hidden,
                    )
                    if config.conditional_prior
                    else LearnableCategoricalPrior(codebook_size=config.vq_codebook_size)
                )
                self.posterior = (
                    LatentPosteriorTrajVQ(
                        state_dim=self.state_dim,
                        z_dim=self.z_dim,
                        hidden_dim=posterior_hidden,
                        n_groups=config.n_groups,
                    )
                    if self.per_episode_z
                    else LatentPosteriorVQ(
                        h_dim=self.h_dim,
                        state_dim=self.state_dim,
                        horizon=config.n_action_steps,
                        z_dim=self.z_dim,
                        hidden_dim=posterior_hidden,
                    )
                )
                from vector_quantize_pytorch import VectorQuantize
                self.vq = VectorQuantize(
                    dim=self.z_dim,
                    codebook_size=config.vq_codebook_size,
                    decay=config.vq_decay,
                    commitment_weight=config.vq_commit_weight,
                )
            else:
                self.prior = (
                    LatentPrior(
                        h_dim=self.h_dim,
                        z_dim=self.z_dim,
                        hidden_dim=prior_hidden,
                        log_sigma_min=config.z_log_sigma_min,
                        log_sigma_max=config.z_log_sigma_max,
                    )
                    if config.conditional_prior
                    else StandardNormalPrior(z_dim=self.z_dim)
                )
                self.posterior = (
                    LatentPosteriorTraj(
                        state_dim=self.state_dim,
                        z_dim=self.z_dim,
                        hidden_dim=posterior_hidden,
                        log_sigma_min=config.z_log_sigma_min,
                        log_sigma_max=config.z_log_sigma_max,
                        n_groups=config.n_groups,
                    )
                    if self.per_episode_z
                    else LatentPosterior(
                        h_dim=self.h_dim,
                        state_dim=self.state_dim,
                        horizon=config.n_action_steps,
                        z_dim=self.z_dim,
                        hidden_dim=posterior_hidden,
                        log_sigma_min=config.z_log_sigma_min,
                        log_sigma_max=config.z_log_sigma_max,
                    )
                )
                self.vq = None

        # Net input is the augmented state (n_obs_steps frames flattened).
        # Cond = concat([h, z]); widens by z_dim (0 if no z).
        self.net = LatentSDEDriftDiffusionNet(
            input_dim=self.state_dim * config.n_obs_steps,
            action_dim=config.action_feature.shape[0],
            cond_dim=self.h_dim + self.z_dim,
            down_dims=config.down_dims,
            n_groups=config.n_groups,
            use_film_scale_modulation=config.use_film_scale_modulation,
            log_sigma_init=config.log_sigma_init,
        )

        if config.compile_model:
            self.net = torch.compile(self.net, mode=config.compile_mode)

        # Per-episode mode pre-caches the normalized state trajectory of every episode in RAM.
        # Populated by `set_train_dataset()` at training start; None otherwise.
        self._episode_state_cache: dict[int, Tensor] | None = None

    def set_train_dataset(self, dataset) -> None:
        """Pre-cache normalized per-episode state trajectories. No-op outside per-episode mode.

        State is MIN_MAX-normalized to [-1, 1] using `dataset.meta.stats['observation.state']`,
        matching the live preprocessor. Memory is negligible (PushT: ~200 KB).
        """
        if not self.per_episode_z:
            return

        stats = dataset.meta.stats.get("observation.state")
        if stats is None or "min" not in stats or "max" not in stats:
            raise ValueError(
                "Per-episode z requires dataset.meta.stats['observation.state'] with 'min'/'max'."
            )
        min_val = torch.as_tensor(stats["min"], dtype=torch.float32).reshape(-1)
        max_val = torch.as_tensor(stats["max"], dtype=torch.float32).reshape(-1)
        span = (max_val - min_val).clamp_min(1e-8)

        # When `cfg.dataset.episodes` subsets the dataset, hf_dataset rows are densified and
        # absolute episode indices must be remapped through the reader.
        ep_indices = dataset.episodes if dataset.episodes is not None else range(dataset.meta.total_episodes)
        abs_to_rel = getattr(getattr(dataset, "reader", None), "_absolute_to_relative_idx", None)

        cache: dict[int, Tensor] = {}
        for ep_idx in ep_indices:
            ep = dataset.meta.episodes[ep_idx]
            abs_rows = list(range(int(ep["dataset_from_index"]), int(ep["dataset_to_index"])))
            rows = abs_rows if abs_to_rel is None else [abs_to_rel[i] for i in abs_rows]
            states = torch.stack(dataset.hf_dataset[rows]["observation.state"]).to(torch.float32)
            cache[ep_idx] = ((2 * (states - min_val) / span - 1)).contiguous()
        self._episode_state_cache = cache

    def _get_episode_state_trajs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Look up cached trajectories for `batch["episode_index"]`; return left-aligned
        `padded` (B, T_max, D) and `valid_mask` (True at in-episode frames)."""
        if self._episode_state_cache is None:
            raise RuntimeError(
                "Per-episode z mode requires set_train_dataset(dataset) before training."
            )
        if "episode_index" not in batch:
            raise KeyError("Per-episode z mode requires 'episode_index' in the batch.")
        ep_indices = batch["episode_index"]
        device = ep_indices.device
        trajs = [self._episode_state_cache[int(i)] for i in ep_indices.reshape(-1).tolist()]
        lengths = torch.tensor([t.shape[0] for t in trajs], dtype=torch.long, device=device)
        padded = nn.utils.rnn.pad_sequence(trajs, batch_first=True).to(device)
        valid_mask = torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]
        return padded, valid_mask

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

    def _augmented_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Flatten the n_obs_steps proprio window into (B, n_obs_steps * state_dim).

        Used on the deployment path (`batch[OBS_STATE]` is the bare n_obs_steps window).
        `compute_loss` builds its own per-tick windows via `unfold` because its
        `batch[OBS_STATE]` covers n_obs_steps + H - 1 frames.
        """
        state = batch[OBS_STATE]
        return state.reshape(state.shape[0], -1)

    def _sde_step(
        self,
        x_now: Tensor,
        mu: Tensor,
        log_sigma: Tensor,
        dt: float,
        deterministic: bool,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        log_sigma = log_sigma.clamp(self.config.log_sigma_min, self.config.log_sigma_max)
        sigma = log_sigma.exp()
        mean = x_now + mu * dt
        std = sigma * (dt ** 0.5)
        if deterministic:
            return mean
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
            return self.vq.get_output_from_indices(k)
        else:
            mu_p, log_sigma_p = self.prior(h)
            if deterministic:
                return mu_p
            eps = torch.randn(mu_p.shape, dtype=mu_p.dtype, device=mu_p.device, generator=generator)
            return mu_p + log_sigma_p.exp() * eps

    def step(
        self,
        x_aug: Tensor,
        h: Tensor,
        z: Tensor | None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """One Euler-Maruyama step from pre-computed h (and optional z).

        Args:
            x_aug: (B, n_obs_steps * state_dim) — augmented state input. The drift/diffusion
                   net reads the full window; the residual update is anchored to the most
                   recent frame, sliced as `x_aug[..., -state_dim:]`.
            noise: (B, action_dim) optional pre-sampled Brownian increment. Mirrors
                   `DiffusionModel.conditional_sample(noise=...)`: when given, replaces the
                   internal `randn`; ignored when `deterministic_inference` is True.
        """
        cond = torch.cat([h, z], dim=-1) if self.use_latent_z else h
        mu, log_sigma = self.net(x_aug, cond)
        dt = self.config.sde_dt if self.config.sde_dt is not None else 1.0
        x_now = x_aug[..., -self.state_dim:]
        return self._sde_step(
            x_now,
            mu,
            log_sigma,
            dt=dt,
            deterministic=self.config.deterministic_inference,
            generator=generator,
            noise=noise,
        )

    def generate_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> Tensor:
        """Single-step action from a self-contained batch; encodes h and samples z fresh.

        The h/z caches live on `LatentSDEPolicy`; this model-layer path stays stateless so
        it's safe for offline eval, mixed-episode batches, and the first tick of a rollout.

        `noise`, when provided, is used as the Euler-Maruyama Brownian increment of the SDE
        step (mirrors `DiffusionModel.generate_actions(noise=...)`). It is *not* used to
        seed the prior z sample — z's stochasticity is governed by the model's RNG.
        """
        h = self.encode_observations(batch)
        z = self.sample_z_from_prior(h)
        x_aug = self._augmented_state(batch)
        return self.step(x_aug, h, z, noise=noise)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Free-space Gaussian NLL on the demo action chunk + β·KL[q||p].

        For demo actions a_0..a_{H-1} starting at time t:
            a_k ~ N(x_seq[k] + μ(x_seq_aug[k], h, z) · dt, diag(σ(...))² · dt)
        x_seq is the measured state trajectory from the dataset (no teacher-forcing),
        so train and inference see the same state distribution.

        Expected `batch` (normalized + on device; LatentSDEPolicy.forward stacks images):
            "observation.state":  (B, n_obs_steps + H - 1, state_dim) — past + future
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
            "action":             (B, H=n_action_steps, action_dim)

        Padding handling: chunks are kept padding-free by `config.drop_n_last_frames`
        at the sampler level, so no loss masking is needed here.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION})
        assert OBS_IMAGES in batch

        action_target = batch[ACTION]                        # (B, H, action_dim)
        B, H, action_dim = action_target.shape
        n_obs = self.config.n_obs_steps
        assert H == self.config.n_action_steps, (
            f"compute_loss expects H == n_action_steps action targets; got H={H} "
            f"vs config.n_action_steps={self.config.n_action_steps}."
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
        x_seq = state_full[:, n_obs - 1 :]  # (B, H, state_dim)
        x_seq_aug = (
            state_full.unfold(dimension=1, size=n_obs, step=1)  # (B, H, D, n_obs)
            .permute(0, 1, 3, 2)                                 # (B, H, n_obs, D)
            .reshape(B, H, n_obs * self.state_dim)
        )

        h = self.encode_observations(batch)                  # (B, cond_dim)

        if self.use_latent_z:
            if self.per_episode_z:
                # Full-episode trajectories from the RAM cache; left-aligned, padded.
                ep_state_traj, valid_mask = self._get_episode_state_trajs(batch)
                # Per-trajectory ELBO: window NLL is an unbiased estimator of the trajectory NLL
                # up to T_ep/H, so scale KL by H/T_ep to match. clamp_min(H) guards degenerate eps.
                ep_lengths = valid_mask.sum(dim=1).to(torch.float32).clamp_min(float(H))
                kl_scale = float(H) / ep_lengths  # (B,)
                posterior_args = (ep_state_traj, valid_mask)
            else:
                kl_scale = 1.0
                posterior_args = (h, x_seq)

            if self.use_vq:
                prior_logits = self.prior(h)
                z_e = self.posterior(*posterior_args)
                z_q_quant, idx_q, _vq_lib_commit_loss = self.vq(z_e.unsqueeze(1))
                z_q = z_q_quant.squeeze(1)
                vq_indices = idx_q.squeeze(1)
                # Per-sample commit & prior-CE so per_episode can apply H/T_ep per element.
                # Commit reproduces the lib's formula (commitment_weight * mse(z_e, sg[z_q])).
                vq_commit_per_sample = (z_e - z_q.detach()).pow(2).mean(dim=-1)  # (B,)
                # k detached on the CE: prior doesn't backprop into posterior/codebook (van den Oord §3.2).
                vq_prior_ce_per_sample = F.cross_entropy(prior_logits, vq_indices.detach(), reduction="none")  # (B,)
                mu_p = log_sigma_p = mu_q = log_sigma_q = None
            else:
                mu_p, log_sigma_p = self.prior(h)
                mu_q, log_sigma_q = self.posterior(*posterior_args)
                eps_z = torch.randn(mu_q.shape, dtype=mu_q.dtype, device=mu_q.device)
                z_q = mu_q + log_sigma_q.exp() * eps_z
        else:
            mu_p = log_sigma_p = mu_q = log_sigma_q = None
            z_q = None

        flat_x_aug = x_seq_aug.reshape(B * H, -1)
        flat_h = h.unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
        if self.use_latent_z:
            flat_z = z_q.unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
            flat_cond = torch.cat([flat_h, flat_z], dim=-1)
        else:
            flat_cond = flat_h
        mu_flat, log_sigma_flat = self.net(flat_x_aug, flat_cond)
        mu = mu_flat.reshape(B, H, action_dim)
        log_sigma = log_sigma_flat.reshape(B, H, action_dim)
        log_sigma = log_sigma.clamp(self.config.log_sigma_min, self.config.log_sigma_max)

        dt = self.config.sde_dt if self.config.sde_dt is not None else 1.0
        dt_t = torch.tensor(dt, device=mu.device, dtype=mu.dtype)
        mean = x_seq + mu * dt
        log_std = log_sigma + 0.5 * torch.log(dt_t)
        var = (2 * log_std).exp()

        nll = 0.5 * ((action_target - mean) ** 2 / var.clamp_min(1e-12) + 2 * log_std)
        nll = nll + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=mu.device, dtype=mu.dtype))

        # Chunks are padding-free via drop_n_last_frames, so a plain mean over batch is correct.
        nll_loss = nll.sum(dim=(1, 2)).mean()

        if self.use_latent_z:
            if self.use_vq:
                vq_commit_loss = (kl_scale * vq_commit_per_sample).mean()
                vq_prior_ce_loss = (kl_scale * vq_prior_ce_per_sample).mean()
                other_loss = self.config.vq_commit_weight * vq_commit_loss + self.config.vq_prior_weight * vq_prior_ce_loss
                # VQ losses are means over latent_dim; scale them up so the final divide
                # by (H * action_dim) doesn't dwarf them next to the NLL.
                other_loss = (H * action_dim) * other_loss
            else:
                kl_per_sample = _gaussian_kl_loss(
                    mu_q, log_sigma_q, mu_p, log_sigma_p, self.config.kl_min
                )  # (B,)
                kl_loss = (kl_scale * kl_per_sample).mean()
                other_loss = self.config.kl_weight * kl_loss
            loss = nll_loss + other_loss
        else:
            loss = nll_loss

        # Final normalization so optimizer/lr scale is independent of H, action_dim.
        loss = loss / (H * action_dim)

        with torch.no_grad():
            loss_dict: dict[str, float] = {
                "nll_loss": nll_loss.detach().item(),
                "action_sigma_mean": log_sigma.exp().mean().item(),
            }
            if self.use_latent_z:
                if self.per_episode_z:
                    loss_dict["ep_length_mean"] = ep_lengths.detach().mean().item()
                if self.use_vq:
                    loss_dict["vq_commit_loss"] = vq_commit_loss.detach().item()
                    loss_dict["vq_prior_ce_loss"] = vq_prior_ce_loss.detach().item()
                    # Perplexity = exp(H(p)) over batch index distribution; max = K.
                    counts = torch.bincount(vq_indices, minlength=self.config.vq_codebook_size).float()
                    probs = counts / counts.sum().clamp_min(1.0)
                    entropy = -(probs * (probs.clamp_min(1e-12)).log()).sum()
                    loss_dict["vq_perplexity"] = entropy.exp().item()
                    loss_dict["vq_active_codes"] = float((counts > 0).sum().item())
                else:
                    loss_dict["kl_loss"] = kl_loss.detach().item()
                    loss_dict["z_sigma_q_mean"] = log_sigma_q.exp().mean().item()
                    loss_dict["z_sigma_p_mean"] = log_sigma_p.exp().mean().item()
        return loss, loss_dict


# Per-episode latent z — prior p(z|h) and amortized posterior q(z|h, x_seq).
# CVAE-style: train z ~ q via reparam, KL[q||p] regularizes the prior. At deployment z is
# resampled from p(z|h) in lock-step with every h refresh, committing each chunk to one mode.
# Conditioning into the drift/diffusion net is cond = concat([h, z], -1).

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
        return zeros, zeros # (μ_p, log σ_p) = (0, 0)


class LatentPrior(nn.Module):
    """p(z | h) — 2-layer MLP producing (μ_p, log σ_p) from the image-only conditioning h."""

    def __init__(
        self,
        h_dim: int,
        z_dim: int,
        hidden_dim: int,
        log_sigma_min: float,
        log_sigma_max: float,
    ):
        super().__init__()
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.trunk = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, z_dim)
        # Wide prior (σ_p ≈ 1) at init so KL doesn't over-constrain q early in training.
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.trunk(h)
        mu = self.mu_head(feat)
        log_sigma = self.log_sigma_head(feat).clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


class LatentPosterior(nn.Module):
    """q(z | h, x_seq) — MLP over [h, flatten(x_seq)].

    Flatten (not pool) over x_seq keeps every step's signal but ties the first Linear's
    in_features to H — the posterior must be rebuilt if `n_action_steps` changes.

    a_seq is deliberately NOT given to q: feeding the target action chunk would let z
    encode it directly, turning the model into a memoriser of demo actions. Restricting
    q to (h, x_seq) forces z to commit a *mode* expressible from the same per-tick
    state + image conditioning the deployed policy sees.
    """

    def __init__(
        self,
        h_dim: int,
        state_dim: int,
        horizon: int,
        z_dim: int,
        hidden_dim: int,
        log_sigma_min: float,
        log_sigma_max: float,
    ):
        super().__init__()
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.horizon = horizon
        traj_summary_dim = horizon * state_dim
        self.trunk = nn.Sequential(
            nn.Linear(h_dim + traj_summary_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, h: Tensor, x_seq: Tensor) -> tuple[Tensor, Tensor]:
        B, H, _ = x_seq.shape
        if H != self.horizon:
            raise ValueError(
                f"LatentPosterior was built with horizon={self.horizon} but got x_seq with H={H}. "
                "The flatten-based posterior is tied to a fixed horizon."
            )
        traj_flat = x_seq.reshape(B, -1)
        feat = self.trunk(torch.cat([h, traj_flat], dim=-1))
        mu = self.mu_head(feat)
        log_sigma = self.log_sigma_head(feat).clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


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


class LatentPosteriorVQ(nn.Module):
    """Deterministic posterior for the VQ path: (h, x_seq) → z_e ∈ ℝ^{z_dim}.

    Mirrors LatentPosterior's trunk with a single linear head (no σ); the quantizer downstream
    snaps z_e to the nearest codebook entry. Horizon-tied via flattened x_seq.
    """

    def __init__(
        self,
        h_dim: int,
        state_dim: int,
        horizon: int,
        z_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.horizon = horizon
        traj_summary_dim = horizon * state_dim
        self.trunk = nn.Sequential(
            nn.Linear(h_dim + traj_summary_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.head = nn.Linear(hidden_dim, z_dim)

    def forward(self, h: Tensor, x_seq: Tensor) -> Tensor:
        B, H, _ = x_seq.shape
        if H != self.horizon:
            raise ValueError(
                f"LatentPosteriorVQ built with horizon={self.horizon} but got x_seq with H={H}. "
                "The flatten-based posterior is tied to a fixed horizon."
            )
        feat = self.trunk(torch.cat([h, x_seq.reshape(B, -1)], dim=-1))
        return self.head(feat)


# ---- Per-episode latent z (config.z_sampling_mode='per_episode') ------------------------------
# Variable-length encoder over the full episode state trajectory. Conv1d + masked GroupNorm +
# masked mean-pool: output at valid positions is bit-equivalent to running on each episode's
# exact-length tensor (pads are zeroed before each Conv1d so kernel-3 doesn't leak across).
class _TrajStateEncoder(nn.Module):
    """Two-layer masked Conv1d encoder + masked mean-pool, returning (B, hidden_dim)."""

    def __init__(self, state_dim: int, hidden_dim: int, n_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(state_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = _MaskedGroupNorm(n_groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = _MaskedGroupNorm(n_groups, hidden_dim)
        self.act = nn.Mish()
        self.post_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish())

    def forward(self, x_traj: Tensor, valid_mask: Tensor) -> Tensor:
        # x_traj: (B, T, state_dim); valid_mask: (B, T) bool.
        m = valid_mask.to(x_traj.dtype).unsqueeze(1)
        h = x_traj.transpose(1, 2) * m                     # zero pads (kernel-3 leak)
        h = self.act(self.norm1(self.conv1(h), valid_mask)) * m
        h = self.act(self.norm2(self.conv2(h), valid_mask))
        return self.post_pool(_masked_mean_pool(h, valid_mask))


class LatentPosteriorTraj(nn.Module):
    """q(z | x_{0:T}) — trajectory encoder → (μ, log σ)."""

    def __init__(
        self,
        state_dim: int,
        z_dim: int,
        hidden_dim: int,
        log_sigma_min: float,
        log_sigma_max: float,
        n_groups: int = 8,
    ):
        super().__init__()
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.encoder = _TrajStateEncoder(state_dim, hidden_dim, n_groups)
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, x_traj: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.encoder(x_traj, valid_mask)
        mu = self.mu_head(feat)
        log_sigma = self.log_sigma_head(feat).clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma


class LatentPosteriorTrajVQ(nn.Module):
    """Deterministic per-episode posterior — trajectory encoder → z_e."""

    def __init__(self, state_dim: int, z_dim: int, hidden_dim: int, n_groups: int = 8):
        super().__init__()
        self.encoder = _TrajStateEncoder(state_dim, hidden_dim, n_groups)
        self.head = nn.Linear(hidden_dim, z_dim)

    def forward(self, x_traj: Tensor, valid_mask: Tensor) -> Tensor:
        return self.head(self.encoder(x_traj, valid_mask))


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
    mu_q: Tensor, log_sigma_q: Tensor, mu_p: Tensor, log_sigma_p: Tensor, kl_min: float = 0.0
) -> Tensor:
    """KL[N(μ_q, diag σ_q²) || N(μ_p, diag σ_p²)] per sample, summed over latent dim.

    kl_min: Per-dim KL floor in nats (free bits). Each dim's KL is clamped to
        max(kl, kl_min) before summing, so the model uses up to kl_min nats
        per dim without KL penalty, preventing posterior collapse.

        For a task with N modes, set kl_min ≈ log(N) / z_dim so the total
        budget z_dim × kl_min ≈ log(N) nats covers the mode bits.
        E.g., N=2 → 0.35 (z_dim=2), 0.04 (z_dim=16). Default 0 disables.
    """
    var_q = (2 * log_sigma_q).exp().clamp_min(1e-12)
    var_p = (2 * log_sigma_p).exp().clamp_min(1e-12)
    per_dim = (log_sigma_p - log_sigma_q) + 0.5 * (var_q + (mu_q - mu_p) ** 2) / var_p - 0.5
    if kl_min > 0.0:
        per_dim = per_dim.clamp_min(kl_min)
    return per_dim.sum(dim=-1)


class LatentSDEDriftDiffusionNet(nn.Module):
    """Point-wise hourglass MLP with per-block FiLM conditioning — port of
    DiffusionConditionalUnet1d (horizon-axis Conv1d → Linear).

    Inputs:
        x:    (B, input_dim)     — augmented state (n_obs_steps frames flattened) so the
                                   net reads local first-differences ≈ velocity.
        cond: (B, cond_dim)      — global conditioning h (or [h, z]).
    Outputs:
        mu:        (B, action_dim) — SDE drift.
        log_sigma: (B, action_dim) — log of per-coordinate diagonal diffusion.

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
        log_sigma_init: float = -2.0,
    ):
        super().__init__()
        assert len(down_dims) >= 1, "`down_dims` must contain at least one width."

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

        # mu: default init (~0). log_sigma: zeros weight + log_sigma_init bias → σ ≈ exp(init)
        # at the start (e.g. log_sigma_init=-2.0 → σ ≈ 0.13 in normalized action space).
        self.mu_head = nn.Linear(widths[-1], action_dim)
        self.log_sigma_head = nn.Linear(widths[-1], action_dim)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, log_sigma_init)

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        feat = x
        for block in self.blocks:
            feat = block(feat, cond)
        feat = self.final_act(self.final_norm(feat.unsqueeze(-1)).squeeze(-1))
        mu = self.mu_head(feat)
        log_sigma = self.log_sigma_head(feat)
        return mu, log_sigma


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
