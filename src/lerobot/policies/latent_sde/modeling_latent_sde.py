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

from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..diffusion.modeling_diffusion import DiffusionRgbEncoder
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_latent_sde import LatentSDEConfig
from .geometry import (
    action_to_endpoint,
    body_to_world,
    local,
    nearest_pose_indices,
    perturb_pose_state,
    pose_from_state,
    pose_state_features,
)
from .modeling_context import ObservationContext


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

    Inference duty cycle (research_brief.md §1.2): the observation context and z (per-episode
    latent, sampled from the prior on that context) are refreshed together every `n_action_steps`
    ticks — matching DP's context-encoder cadence. The light drift/diffusion net runs every tick
    on the current measured state and z only.
    """

    config_class = LatentSDEConfig
    name = "latent_sde"

    def __init__(self, config: LatentSDEConfig, **kwargs):
        super().__init__(config)
        self.config = config
        # Training from scratch: sweep the dataset once for the action scale (cached, stored in the config).
        if config.action_scale is None and not config.pretrained_path and kwargs.get("dataset_meta") is not None:
            from .action_scale import resolve_action_scale

            config.action_scale = resolve_action_scale(config, kwargs["dataset_meta"])

        self._queues = None
        self._steps_until_refresh: int = 0
        self._cached_z: Tensor | None = None
        self._cached_x0: Tensor | None = None

        self.model = LatentSDEModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """Clear observation queues and the cached z. Call on `env.reset()`."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        self._steps_until_refresh = 0
        self._cached_z = None
        self._cached_x0 = None

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

        Re-encode the context and re-sample z every `n_action_steps` ticks (chunk-local hold).
        """
        batch = dict(batch)
        batch.pop(ACTION, None)

        camera_keys = self.config.image_features
        if camera_keys:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in camera_keys], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        # The fast drift input is the current state; the context uses the causal observation window.
        x_now = self._queues[OBS_STATE][-1]                                # (B, D) — current state frame

        # Slow path: re-encode the context and re-sample z every n_action_steps ticks.
        if self._cached_z is None or self._steps_until_refresh == 0:
            stacked_images = torch.stack(list(self._queues[OBS_IMAGES]), dim=1)
            stacked_states = torch.stack(list(self._queues[OBS_STATE]), dim=1)
            context_batch = {OBS_IMAGES: stacked_images, OBS_STATE: stacked_states}
            if self.config.context_encoder == "smolvlm2":
                for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
                    context_batch[key] = batch[key]
            self._cached_z = self.model.sample_z_from_prior(self.model.encode_observations(context_batch))
            # Chunk-initial state: the origin for normalize_state (held for the whole refresh window).
            self._cached_x0 = x_now.clone()
            self._steps_until_refresh = self.config.n_action_steps

        action = self.model.step(x_now, self._cached_z, x0=self._cached_x0, noise=noise)
        self._steps_until_refresh -= 1
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float] | None]:
        """Run the batch through the model and compute the loss for training or validation."""
        camera_keys = self.config.image_features
        if camera_keys:
            batch = dict(batch)
            for key in camera_keys:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in camera_keys], dim=-4)
        loss, loss_dict = self.model.compute_loss(batch)
        return loss, loss_dict


class LatentSDEModel(nn.Module):
    """Assembles h, holds drift/diffusion net, exposes sample + compute_loss.

    Mirrors the role of DiffusionModel one-for-one.
    """

    def __init__(self, config: LatentSDEConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]
        if config.sde_geometry not in ("euclidean", "so3_r3_body"):
            raise ValueError(f"Unsupported SDE geometry {config.sde_geometry!r}.")
        self.pose_geometry = config.sde_geometry == "so3_r3_body"
        self.state_feature_dim = 11 if self.pose_geometry else self.state_dim

        # Only the prior reads the joint observation context; the drift reads z and current proprioception.
        global_cond_dim = 0
        num_images = len(self.config.image_features)
        if self.config.context_encoder == "smolvlm2":
            from .modeling_smolvlm_context import SmolVLMContextEncoder

            self.context_encoder = SmolVLMContextEncoder(config, state_dim=self.state_feature_dim)
            latent_hidden_dim = 512
        elif self.config.use_separate_rgb_encoder_per_camera:
            # DiffusionRgbEncoder reads a few fields from the config; LatentSDEConfig matches
            # DiffusionConfig's names so no adaptation is needed.
            encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
            self.rgb_encoder = nn.ModuleList(encoders)
            global_cond_dim += encoders[0].feature_dim * num_images
        else:
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += self.rgb_encoder.feature_dim * num_images

        if self.config.context_encoder == "resnet":
            latent_hidden_dim = global_cond_dim * config.n_obs_steps
            self.state_encoder = nn.Sequential(
                nn.Linear(config.n_obs_steps * self.state_feature_dim, latent_hidden_dim),
                nn.Mish(),
            )
            self.h_dim = 2 * latent_hidden_dim

        # p(z|context) and an action-only posterior.
        self.use_vq = config.use_vq

        self.normalize_state = config.normalize_state

        self.z_dim = config.z_dim
        posterior_hidden = config.z_posterior_hidden_dim or latent_hidden_dim
        prior_hidden = config.z_prior_hidden_dim or latent_hidden_dim

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
            self.prior = LatentPriorVQ(
                h_dim=self.h_dim,
                codebook_size=self.num_codes,
                hidden_dim=prior_hidden,
            )
            self.posterior = LatentPosteriorTrajVQ(
                input_dim=self.action_dim,
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
            if config.context_encoder == "smolvlm2":
                from .modeling_token_kv import SmolVLMTokenKVPrior

                self.prior = SmolVLMTokenKVPrior(
                    config,
                    self.context_encoder.text_config,
                    z_dim=self.z_dim,
                    sigma_act=_sigma_act(config.sigma_activation),
                    sigma_min=config.z_sigma_min,
                )
            else:
                self.prior = LatentPrior(
                    h_dim=self.h_dim,
                    z_dim=self.z_dim,
                    hidden_dim=prior_hidden,
                    sigma_activation=config.sigma_activation,
                    sigma_min=config.z_sigma_min,
                )
            self.posterior = LatentPosteriorTraj(
                input_dim=self.action_dim,
                z_dim=self.z_dim,
                hidden_dim=posterior_hidden,
                sigma_activation=config.sigma_activation,
                sigma_min=config.z_sigma_min,
                num_levels=tcn_levels,
                n_groups=config.n_groups,
            )
            self.vq = None

        # Light per-tick drift: current state features in, z as the only FiLM conditioning.
        self.net = LatentSDEDriftDiffusionNet(
            input_dim=self.state_feature_dim,
            action_dim=config.action_feature.shape[0],
            cond_dim=self.z_dim,
            down_dims=config.down_dims,
            n_groups=config.n_groups,
            use_film_scale_modulation=config.use_film_scale_modulation,
        )

        # Action-decoder variance σ² (SDE diffusion coeff²): a buffer, NOT gradient-trained — EMA'd
        # toward the analytic per-batch MLE mean‖d*−μ‖² (calibrated σ-VAE, arXiv:2006.13202), and
        # WARM-STARTED from the first training batch. Warm-start matters: a σ²=1 init down-weights recon
        # by ~1/σ² and stalls early convergence (posterior collapses under β before σ calibrates). Feeds
        # the Gaussian NLL and SDE noise σ. One scalar in Euclidean mode; body mode keeps one σ² per
        # action-scale group (position, rotation, gripper), whose residuals differ in kind.
        # Euclidean VQ keeps σ²=1.
        self.register_buffer("action_var", torch.ones((3,) if self.pose_geometry else ()))
        self.register_buffer("sigma_initialized", torch.zeros((), dtype=torch.bool))
        if self.pose_geometry:
            # Output coordinate → σ² group.
            self.register_buffer("sigma_group", torch.tensor([0, 0, 0, 1, 1, 1, 2]), persistent=False)
        # Per-dim action scale s (config.action_scale): drift target d/s, output s·μ.
        scale = config.action_scale if config.action_scale is not None else [1.0] * self.action_dim
        if len(scale) != self.action_dim:
            raise ValueError(f"`action_scale` needs {self.action_dim} values, got {len(scale)}.")
        self.register_buffer("action_scale", torch.tensor(scale, dtype=torch.float32), persistent=False)

        if config.compile_model:
            self.net = torch.compile(self.net, mode=config.compile_mode)

    def encode_observations(self, batch: dict[str, Tensor]) -> ObservationContext:
        """Encode the causal image/state window and optional task into joint context.

        Slow path: runs the context encoder over `n_obs_steps` stacked frames. At deployment,
        the result is cached and reused for `n_action_steps` ticks (matches DP's context-encoder
        duty cycle).

        Returns the prior's input: ResNet h of shape (B, h_dim), or SmolVLM2's per-layer prefix K/V.
        """
        state = batch[OBS_STATE]
        state_obs = self._state_features(state[:, : self.config.n_obs_steps])
        if self.config.context_encoder == "smolvlm2":
            return self.context_encoder(
                batch[OBS_IMAGES],
                batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK],
                state_obs,
            )

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

        state_features = self.state_encoder(state_obs.flatten(start_dim=1))
        return ObservationContext(h=torch.cat([img_features.flatten(start_dim=1), state_features], dim=-1))

    def _dim_var(self) -> Tensor:
        """σ² per output coordinate (its group's value in body mode, the scalar in Euclidean mode)."""
        return self.action_var[self.sigma_group] if self.pose_geometry else self.action_var

    def _state_features(self, state: Tensor, reference_state: Tensor | None = None) -> Tensor:
        if self.pose_geometry:
            return pose_state_features(state, reference_state)
        return state - reference_state if reference_state is not None else state

    def _prepare_pose_training(
        self,
        state: Tensor,
        action: Tensor,
        action_is_pad: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, horizon = state.shape[:2]
        augment = self.training and self.config.state_noise_std > 0
        if action_is_pad is None:
            if self.config.do_mask_loss_for_padding or (augment and self.config.action_anchor == "nearest"):
                raise ValueError("Pose masking or nearest augmentation requires 'action_is_pad'.")
            candidates = torch.ones(batch_size, horizon, dtype=torch.bool, device=state.device)
        else:
            candidates = ~action_is_pad
        if batch_size == 0 or not candidates.any(dim=-1).all():
            raise ValueError("Each pose training chunk must have at least one valid action.")

        if self.config.do_mask_loss_for_padding:
            state = torch.where(candidates[..., None], state, torch.zeros_like(state))
            action = torch.where(candidates[..., None], action, torch.zeros_like(action))
        pose = pose_from_state(state)
        target_pose = action_to_endpoint(pose, action[..., :6])
        target_gripper = action[..., 6:7]
        # Posterior input: the clean body-frame increment / s (the noise-free drift target).
        post_action = torch.cat((local(pose, target_pose), target_gripper), dim=-1) / self.action_scale.to(state.dtype)
        query = state
        if augment:
            std: float | Tensor = self.config.state_noise_std * self.action_scale[:6].to(state.dtype)
            if self.config.state_noise_schedule == "linear":
                ramp = torch.arange(1, horizon + 1, device=state.device, dtype=state.dtype) / horizon
                std = std * ramp[None, :, None]
            query = perturb_pose_state(state, torch.randn_like(state[..., :6]) * std)
            if self.config.do_mask_loss_for_padding:
                query = torch.where(candidates[..., None], query, state)
            if self.config.action_anchor == "nearest":
                indices = nearest_pose_indices(query, state, candidates)
                target_pose, target_gripper = (
                    value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))
                    for value in (target_pose, target_gripper)
                )
        tangent = local(pose_from_state(query), target_pose)
        return state, post_action, query, torch.cat((tangent, target_gripper), dim=-1), candidates

    def _sde_step(
        self,
        x_now: Tensor,
        mu: Tensor,
        deterministic: bool,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        # One-step action = s·(μ + σ·ε).
        delta = mu
        if not deterministic:
            std = self._dim_var().sqrt().to(delta.dtype)
            if noise is not None:
                eps = noise.to(dtype=delta.dtype, device=delta.device)
            else:
                eps = torch.randn(delta.shape, dtype=delta.dtype, device=delta.device, generator=generator)
            delta = delta + std * eps
        return x_now + self.action_scale.to(delta.dtype) * delta

    def _prior(self, context: ObservationContext):
        """Prior head on the observation context: SmolVLM2's reads the layer K/V, ResNet's reads h."""
        return self.prior(context) if self.config.context_encoder == "smolvlm2" else self.prior(context.h)

    def sample_z_from_prior(
        self,
        context: ObservationContext,
        deterministic: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample z from the joint-context prior."""
        if deterministic is None:
            deterministic = self.config.deterministic_z_inference
        if self.use_vq:
            logits = self._prior(context)
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
            mu_p, sigma_p = self._prior(context)
            if deterministic:
                return mu_p
            eps = torch.randn(mu_p.shape, dtype=mu_p.dtype, device=mu_p.device, generator=generator)
            return mu_p + sigma_p * eps

    def predict_drift(self, state: Tensor, z: Tensor, *, x0: Tensor | None = None) -> Tensor:
        """Evaluate raw (B,H,D) states under FiLM(z), optionally relative to the clean (B,1,D) chunk origin."""
        if self.normalize_state and x0 is None:
            raise ValueError("normalize_state=True requires the clean chunk-initial state x0.")
        state = self._state_features(state, x0 if self.normalize_state else None)
        batch_size, horizon = state.shape[:2]
        flat_z = z[:, None].expand(batch_size, horizon, -1).reshape(batch_size * horizon, -1)
        return self.net(state.reshape(batch_size * horizon, -1), flat_z).reshape(batch_size, horizon, -1)

    def step(
        self,
        x_now: Tensor,
        z: Tensor,
        x0: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """One Euclidean or body-tangent Gaussian step from the cached z.

        Args:
            x_now: (B, state_dim) — the CURRENT state frame. The drift net reads only this single
                   frame (velocity-blind); the residual update is anchored to it.
            x0: (B, state_dim) clean chunk-initial state, required when normalize_state=True.
                Euclidean drift inputs subtract x0; pose inputs use its position and rotation as a
                fixed reference frame. Integration and controller conversion still use absolute x_now.
            noise: (B, action_dim) optional standardized noise. Mirrors
                   `DiffusionModel.conditional_sample(noise=...)`: when given, replaces the
                   internal `randn`; ignored when `deterministic_inference` is True.
                   Body mode uses ordinary Gaussian noise in normalized body-controller coordinates.
        """
        mu = self.predict_drift(x_now[:, None], z, x0=x0[:, None] if x0 is not None else None)[:, 0]
        if self.pose_geometry:
            mu = mu.to(torch.float64 if x_now.dtype == torch.float64 else torch.float32)
        # Keep controller actions in the measured-state dtype under autocast.
        integration_anchor = x_now.new_zeros(mu.shape) if self.pose_geometry else x_now
        action = self._sde_step(
            integration_anchor,
            mu,
            deterministic=self.config.deterministic_inference,
            generator=generator,
            noise=noise,
        )
        if self.pose_geometry:
            # Equivalent to body retraction at the controller boundary, without an Exp/Log round
            # trip that could turn an exact zero into an OSC goal-changing epsilon command.
            world = body_to_world(pose_from_state(x_now), action[..., :6])
            action = torch.cat((world, action[..., 6:7]), dim=-1)
            return action.clamp(-1, 1)
        return action

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """ELBO loss (Gaussian path): nll + beta·KL[q‖p].

        In Euclidean mode, nll = Gaussian NLL of the one-step decoder d*/s ~ N(μ, σ²): per element
        0.5·log(2πσ²) + (d*−μ)²/(2σ²), where d* = a−x (target-state actions) divided by the action
        scale s. Terms are summed over the H·D steps then /(H·D);
        KL is likewise /(H·D), so both share the scale and β keeps its meaning (β=1 = ELBO). σ²
        (`self.action_var`) is NOT gradient-trained — an EMA of the per-batch MLE mean‖d*−μ‖²
        (σ-VAE), so the NLL trains only μ. The optimized loss is (nll + β·KL)·sg(2σ̄²), σ̄² the mean σ²
        over coordinates (= σ² in Euclidean mode, whose recon gradient is then the plain MSE gradient).
        VQ/FSQ keep MSE-mean recon + prior-CE (+ commitment for
        "vq"); a plain-MSE `recon_loss` is logged for the z-usage/leakage diagnostics.

        Body mode uses local(query, nominal endpoint), expressed in controller units, as the delta
        target, with one σ² EMA per position/rotation/gripper group. The same H*7 normalization applies.
        This is a local tangent approximation, not a globally normalized manifold transition.
        Nearest augmentation relabels endpoints, not the clean posterior actions.

        x_seq is the measured state trajectory from the dataset (no teacher-forcing). Train and
        inference see the same state distribution only when state_noise_std==0; under state-noise the
        drift input is perturbed at train time (corrective augmentation), while inference stays clean.
        The posterior reads only the demonstration actions.

        Expected `batch` (normalized + on device; LatentSDEPolicy.forward stacks images):
            "observation.state":  (B, n_obs_steps-1+horizon, state_dim) — causal context + horizon
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
            "action":             (B, horizon, action_dim)
            "action_is_pad":      (B, horizon) — used iff do_mask_loss_for_padding

        Padding handling: `drop_n_last_frames` keeps the EXECUTED region unpadded, but the
        predicted tail may be copy-padded at episode ends. When do_mask_loss_for_padding=True,
        those ticks are masked out of the recon MSE and the posterior valid_mask
        (`valid = ~action_is_pad`); otherwise they are included unmasked (DP-style default).
        """
        action_target = batch[ACTION]                        # (B, H, action_dim)
        B, H, action_dim = action_target.shape

        # Leading n_obs frames form causal context; trailing H frames supervise the pointwise drift.
        state_full = batch[OBS_STATE]
        n_lead = self.config.n_obs_steps - 1
        state_seq = state_full[:, n_lead:]                    # (B, H, state_dim) — horizon states (clean)
        noise = self.training and self.config.state_noise_std > 0
        std = self.config.state_noise_std * self.action_scale.to(state_seq.dtype)

        # Drift input (train-only state-noise): perturb the demo states, then recompute the recon
        # target from the perturbed anchor (corrective drift). The posterior still reads clean actions.
        # state_noise_schedule: "uniform" = std·s on every tick; "linear" ramps it (1/H → 1)× over the chunk.
        pose_drift_target = None
        pose_candidates = None
        if self.pose_geometry:
            # action_target becomes the posterior input: clean body-frame increment / s.
            state_seq, action_target, state_win, pose_drift_target, pose_candidates = self._prepare_pose_training(
                state_seq, action_target, batch.get("action_is_pad")
            )
        elif not noise:
            state_win = state_seq
        elif self.config.state_noise_schedule == "linear":
            delta = torch.arange(H, device=state_seq.device, dtype=state_seq.dtype)
            std_t = std * ((delta + 1) / H).view(-1, 1)  # (H, D) std/H at tick 0 → std at tick H-1
            state_win = state_seq + std_t.unsqueeze(0) * torch.randn_like(state_seq)
        else:  # "uniform"
            state_win = state_seq + std * torch.randn_like(state_seq)

        x_seq = state_win                                    # (B, H, state_dim) — noised anchor; drift's frame
        x_seq_clean = state_seq                              # (B, H, state_dim) — clean demo anchor x_k

        # Hold the clean chunk origin, not the augmented query or the oldest context observation.
        # Only drift features and Euclidean posterior actions are re-centered; targets stay unchanged.
        x0 = state_seq[:, :1] if self.normalize_state else None  # (B, 1, state_dim)

        context = self.encode_observations(batch)

        # Padding mask (DP-style). `action_is_pad` (B, H) marks copy-padded chunk ticks at episode
        # ends; it aligns with the action/state deltas 0..H-1. Tick k's recon uses action_target[:, k]
        # and x_seq[:, k] (both delta k), and the posterior entry at tick k is delta k too, so the
        # per-tick mask is valid[:, k]. Off → all-valid (identical to the legacy no-mask behavior).
        if self.pose_geometry:
            valid = (
                pose_candidates
                if self.config.do_mask_loss_for_padding
                else torch.ones(B, H, dtype=torch.bool, device=x_seq.device)
            )
        elif self.config.do_mask_loss_for_padding:
            valid = ~batch["action_is_pad"]                  # (B, H) True = real frame
        else:
            valid = torch.ones(B, H, dtype=torch.bool, device=x_seq.device)

        # Pose mode already holds body-frame increments; only Euclidean target states are re-centered.
        post_action = action_target
        if x0 is not None and not self.pose_geometry:
            post_action = action_target - x0

        if self.use_vq:
            prior_logits = self._prior(context)
            z_e = self.posterior(post_action, valid)
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
            mu_p, sigma_p = self._prior(context)
            mu_q, sigma_q = self.posterior(post_action, valid)
            eps_z = torch.randn(mu_q.shape, dtype=mu_q.dtype, device=mu_q.device)
            z_q = mu_q + sigma_q * eps_z

        mu = self.predict_drift(x_seq, z_q, x0=x0)

        if self.pose_geometry:
            assert pose_drift_target is not None
            drift_target = pose_drift_target
        else:
            # Recon target d* = a_anchor − x̃: one-step full return to a demo action from the noised
            # anchor x̃ = x_seq. action_anchor picks a_anchor:
            #   "clean"   — the corresponding-index action a_k (== legacy corrective target).
            #   "nearest" — the action a_j of the nearest demo state x_j over the chunk (autonomous field; z
            #               resolves the branch ambiguity at self-intersections). Equals "clean" with no tube.
            if self.config.action_anchor == "nearest":
                nn_idx = torch.cdist(x_seq, x_seq_clean).argmin(dim=-1)  # (B,H) nearest clean-state index
                a_anchor = torch.gather(action_target, 1, nn_idx.unsqueeze(-1).expand(-1, -1, action_dim))
            else:  # "clean"
                a_anchor = action_target
            drift_target = a_anchor - x_seq
        drift_target = drift_target / self.action_scale.to(drift_target.dtype)
        sq_err = (mu - drift_target) ** 2                 # (B, H, action_dim) — unmasked
        # Plain-MSE recon (zero copy-padded ticks, normalize by NOMINAL B·H·D, reduces to a plain mean
        # when nothing is masked). Not the training objective in the Gaussian path — kept for logging
        # and the z-usage / prior-leakage diagnostics below, which compare recon MSE across z choices.
        masked_se = sq_err * valid.unsqueeze(-1) if self.config.do_mask_loss_for_padding else sq_err
        recon_loss = masked_se.mean()

        if not self.use_vq or self.pose_geometry:
            if self.pose_geometry:
                # Pool sufficient statistics, not rank-local means: valid counts may differ.
                error = sq_err.detach().to(
                    torch.float64 if sq_err.dtype == torch.float64 else torch.float32
                )
                dim_sums = (error * valid.unsqueeze(-1)).sum(dim=(0, 1))
                sums = dim_sums.new_zeros(3).index_add_(0, self.sigma_group, dim_sums)
                counts = valid.sum().to(sums) * torch.bincount(self.sigma_group).to(sums)
                stats = torch.cat((sums, counts))
                if (
                    self.training
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.all_reduce(stats)
                batch_var = (stats[:3] / stats[3:]).clamp_min(1e-8)
            elif self.config.do_mask_loss_for_padding:
                batch_var = (sq_err.detach() * valid.unsqueeze(-1)).sum() / (valid.sum() * action_dim).clamp_min(1)
            else:
                batch_var = sq_err.detach().mean()
            if self.training:
                d = self.config.sigma_ema_decay
                with torch.no_grad():
                    if self.sigma_initialized:
                        self.action_var.mul_(d).add_((1.0 - d) * batch_var)
                    else:
                        self.action_var.copy_(batch_var)
                        self.sigma_initialized.fill_(True)

        nll_loss = None
        if self.use_vq:
            # VQ/FSQ regression in decoder coordinates + commitment/prior-CE regularizer.
            vq_prior_ce_loss = vq_prior_ce_per_sample.mean()
            if self.config.quantizer == "vq":
                # vq_commit_loss already carries commitment_weight (applied inside VectorQuantize).
                other_loss = vq_commit_loss + self.config.vq_prior_weight * vq_prior_ce_loss
            else:  # "fsq" — no commitment loss, only prior-CE
                other_loss = self.config.fsq_prior_weight * vq_prior_ce_loss
            loss = recon_loss + other_loss
        else:
            var = self._dim_var()                            # σ² — detached buffer, so only μ gets a gradient
            nll_elem = 0.5 * math.log(2 * math.pi) + 0.5 * var.log() + sq_err / (2 * var)
            if self.config.do_mask_loss_for_padding:
                nll_elem = nll_elem * valid.unsqueeze(-1)    # padded ticks are not observations
            # nll (sum over H·D) and KL (sum over z_dim) both /(H·D): shrinks magnitude, β meaning kept.
            norm = H * action_dim
            nll_loss = nll_elem.sum(dim=(1, 2)).mean() / norm
            kl_per_sample = _gaussian_kl_loss(mu_q, sigma_q, mu_p, sigma_p)  # (B,)
            kl_loss = kl_per_sample.mean() / norm
            loss = nll_loss + self.config.beta * kl_loss  # β-VAE ELBO
            # × sg(2σ̄²), σ̄² the mean over coordinates: the recon gradient is the MSE gradient weighted by
            # σ̄²/σ² per coordinate (plain MSE with one σ²); β ratio unchanged.
            loss = loss * (2.0 * var.mean())

        with torch.no_grad():
            loss_dict: dict[str, float] = {
                "recon_loss": recon_loss.detach().item(),
                "effective_sigma": self._dim_var().mean().sqrt().item(),
                "target_rms": (
                    (drift_target.detach().pow(2) * valid.unsqueeze(-1)).sum() / (valid.sum() * action_dim)
                ).sqrt().item(),
            }
            if self.pose_geometry:
                loss_dict.update(zip(("sigma_pos", "sigma_rot", "sigma_grip"), self.action_var.sqrt().tolist()))
            if nll_loss is not None:
                loss_dict["nll_loss"] = nll_loss.detach().item()
            # z_usage_gap: extra recon error from a batch-rolled (mismatched) z. ~0 ⇒ z ignored.
            mu_rolled = self.predict_drift(x_seq, torch.roll(z_q, shifts=1, dims=0), x0=x0)
            rolled_se = (mu_rolled - drift_target) ** 2
            if self.config.do_mask_loss_for_padding:
                rolled_se = rolled_se * valid.unsqueeze(-1)
            recon_loss_rolled = rolled_se.mean()
            loss_dict["z_usage_gap"] = (recon_loss_rolled - recon_loss).item()
            # recon_loss_prior: recon with z ~ p(z|context) — the DEPLOY-time z (training uses q). The
            # posterior's recon gain transfers to deployment only if this stays near recon_loss; if
            # it rises toward the rolled (mismatched-z) level, the gain is posterior LEAKAGE — z
            # encodes trajectory info the prior can't reproduce. prior_recon_gap = the deploy penalty.
            z_prior = self.sample_z_from_prior(context)   # (B, z_dim), deploy prior distribution
            mu_prior = self.predict_drift(x_seq, z_prior, x0=x0)
            prior_se = (mu_prior - drift_target) ** 2
            if self.config.do_mask_loss_for_padding:
                prior_se = prior_se * valid.unsqueeze(-1)
            recon_loss_prior = prior_se.mean()
            loss_dict["recon_loss_prior"] = recon_loss_prior.item()
            loss_dict["prior_recon_gap"] = (recon_loss_prior - recon_loss).item()
            if self.use_vq:
                loss_dict["other_loss"] = other_loss.detach().item()
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


# Per-episode latent z — joint-context prior p(z|context) and action-trajectory posterior q(z|a_seq).
# CVAE-style: train z ~ q via reparam, KL[q||p] regularizes the prior. At deployment z is
# resampled from the prior in lock-step with every context refresh, committing each chunk to one mode.
# z is the drift net's only FiLM conditioning; the net input is the current state features.

class LatentPrior(nn.Module):
    """p(z | h) — 2-layer MLP producing (mu_p, sigma_p) from joint observation context."""

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


class LatentPriorVQ(nn.Module):
    """p(k | h) over codebook indices from joint context; trained against detached code indices.

    Head init zeros gives a uniform prior at initialization.
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
# action trajectories — 1×1 lift + kernel-3 dilated residual blocks + masked mean-pool. Pads
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
        # traj: (B, T, action_dim); valid_mask: (B, T) bool.
        m = valid_mask.to(traj.dtype).unsqueeze(1)
        h = self.input_proj(traj.transpose(1, 2) * m)
        for block in self.blocks:
            h = block(h, valid_mask, m)
        return self.post_pool(_masked_mean_pool(h, valid_mask))


class LatentPosteriorTraj(nn.Module):
    """q(z | a_{0:T}) — action-trajectory encoder, MLP trunk, and Gaussian parameter heads."""

    def __init__(
        self,
        input_dim: int,
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
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.mu_head = nn.Linear(hidden_dim, z_dim)
        self.sigma_head = nn.Linear(hidden_dim, z_dim)
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

    def forward(self, traj: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.encoder(traj, valid_mask)
        feat = self.trunk(feat)
        mu = self.mu_head(feat)
        sigma = self.sigma_act(self.sigma_head(feat)) + self.sigma_min
        return mu, sigma


class LatentPosteriorTrajVQ(nn.Module):
    """Deterministic action-trajectory posterior with an MLP trunk and z_e head."""

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        hidden_dim: int,
        num_levels: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.encoder = _TrajEncoder(input_dim, hidden_dim, num_levels, n_groups)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.head = nn.Linear(hidden_dim, z_dim)

    def forward(self, traj: Tensor, valid_mask: Tensor) -> Tensor:
        feat = self.encoder(traj, valid_mask)
        return self.head(self.trunk(feat))


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


def _gaussian_kl_loss(mu_q: Tensor, sigma_q: Tensor, mu_p: Tensor, sigma_p: Tensor) -> Tensor:
    """KL[N(μ_q, diag σ_q²) || N(μ_p, diag σ_p²)] per sample, summed over latent dim."""
    var_q = sigma_q.pow(2)
    var_p = sigma_p.pow(2)
    per_dim = sigma_p.log() - sigma_q.log() + 0.5 * (var_q + (mu_q - mu_p) ** 2) / var_p - 0.5
    return per_dim.sum(dim=-1)


class LatentSDEDriftDiffusionNet(nn.Module):
    """Point-wise hourglass MLP with per-block FiLM conditioning — port of
    DiffusionConditionalUnet1d (horizon-axis Conv1d → Linear).

    Inputs:
        x:    (B, input_dim)     — current state features.
        cond: (B, cond_dim)      — the latent z (FiLM); the drift never reads observation context.
    Output:
        mu:   (B, action_dim) — SDE drift. The action-decoder σ is a calibrated EMA buffer on
                                LatentSDEModel (action_var = σ²); inference noise = s·σ·ε.

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
    ):
        super().__init__()

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

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        feat = x
        for block in self.blocks:
            feat = block(feat, cond)
        feat = self.final_act(self.final_norm(feat.unsqueeze(-1)).squeeze(-1))
        return self.mu_head(feat)


class FiLMResidualMLPBlock(nn.Module):
    """Point-wise ResNet block with FiLM-with-scale conditioning.

    Mirrors DiffusionConditionalResidualBlock1d (Conv1d → Linear since the SDE acts on a
    single time step), with scale/bias a plain Linear of cond:
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
        # Linear FiLM generator: z is already a learned code, and DP's leading Mish would alias z < -1.19
        # (Mish is not injective there).
        self.cond_encoder = nn.Linear(cond_dim, cond_channels)

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
