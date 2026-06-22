#!/usr/bin/env python
#
# LatentSDE Policy — configuration for
# "Latent-SDE Policies for Hierarchical Robot Manipulation" (research_brief.md v7).
#
# PoC scope:
#   * per-episode latent strategy z with prior p_ψ(z|h) and posterior q_φ(z|h, x_seq, a_seq);
#     drift/diffusion conditioned on cond = concat([h, z], -1);
#   * free-space Euler-Maruyama log-likelihood (research_brief.md §3.7) + KL[q||p] (β-VAE).
#
# Deferred: controller-pushforward objective and (M, K) compliance heads (Tier 3).
#
# Fairness vs. DiffusionPolicy on Push-T: same vision backbone (DiffusionRgbEncoder),
# same FiLM-with-scale conditioning, same GroupNorm n_groups, same down_dims width ladder.
# Only difference: the chunk-horizon Conv1d collapses to point-wise Linear because the SDE
# is integrated one step at a time on the measured state x.

import logging
from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("latent_sde")
@dataclass
class LatentSDEConfig(PreTrainedConfig):
    """Configuration class for LatentSDEPolicy.

    Defaults are tuned for Push-T (proprio + single camera) and mirror DiffusionPolicy
    so this PoC is a like-for-like replacement of the denoising U-Net.

    SDE roles (research_brief.md §1.2, §3):
        x  — measured robot state (proprio). Push-T: 2-D `observation.state` (agent_pos).
             The drift/diffusion net reads **all n_obs_steps frames flattened** (augmented
             input, mirrors DP's state branch) so it has access to first-differences ≈
             velocity; the Euler-Maruyama residual is still anchored to the most recent
             frame (`mean = s_t + μ·dt`). Fed at every Tier-2 tick. At training time the
             chunk's x_seq is sampled directly from the dataset (see
             `observation_delta_indices_per_key`), matching the deployment-time stream.
        h  — perception conditioning (Tier-1). Image features only in this PoC: vision
             encoder output stacked over `n_obs_steps` frames, flattened, fed via FiLM.
             Refreshed every `n_action_steps` ticks so the vision-encoder duty cycle
             matches DiffusionPolicy's (fair compute) and the Tier-1/Tier-2 rate split
             is reproduced architecturally. cf. notes/h_is_conditioning.tex
        z  — per-episode latent strategy. CVAE-style: prior p(z|h) is re-sampled at
             deployment **in lock-step with every h refresh** ("episode" =
             one h-refresh window), committing each chunk to one mode. At training,
             posterior q(z|h, x_seq, a_seq) provides chunk-level mode signal; loss = NLL +
             kl_weight · KL[q||p]. Conditioning is cond = concat([h, z], -1); the
             drift/diffusion block structure is unchanged.

    `observation.environment_state` (e.g. Push-T's 16-D T-block pose) is currently ignored
    to keep the h-is-image-only contract clean. Add a concat-into-x or concat-into-h route
    if the experiment warrants it.

    Drift/diffusion network output:
        mu — SDE drift only, shape (B, action_dim). Action σ is NOT learned.
        SDE step (inference): x_d = x + mu·dt + σ_eff·√dt·ε, with σ_eff = √(kl_weight/2)
        the SDE diffusion coefficient. Training loss: recon = mean‖μ − v*‖² +
        (kl_weight/(H·action_dim·dt)) · KL[q‖p] (β-VAE form) where v* = (a − x)/dt is the
        empirical one-step velocity; the /(H·action_dim·dt) KL scaling is ELBO-exact under
        the SDE decoder Δx ~ N(μ·dt, σ²·dt), so kl_weight = 2·σ_eff² holds exactly.

    Push-T I/O (mirrors DiffusionConfig):
        - "observation.state" required.
        - At least one "observation.image*" key required.
        - "action" required. For Push-T, action == next end-effector pose target,
          making the kinematic-imitation assumption x_d ≈ x exact in form.

    New / different args vs. DiffusionConfig:
        n_action_steps:   image-conditioning refresh period at deployment (same semantics
                          as DP). Also the SDE chunk length at training time: per sample,
                          h is encoded once and the SDE is unrolled H=n_action_steps steps
                          under teacher-forced demo actions — same per-image-encode
                          supervision budget as DP's horizon-length chunk loss.
        sde_dt:           Δt for one Euler-Maruyama step. Push-T fps=10 Hz → 0.1 s.
        sigma_activation: "exp" or "softplus"; used only by z prior/posterior σ heads.
        kl_weight:        β on KL[q||p], divided by (H·action_dim·dt) in the loss for
                          ELBO-exact balance: kl_weight = 2·σ_eff² exactly (σ_eff = SDE
                          diffusion coefficient). Inference SDE noise uses σ_eff = √(kl_weight/2)

    Removed (no analog in single-step SDE):
        horizon, noise scheduler block, diffusion_step_embed_dim,
        num_inference_steps, kernel_size, clip_sample*.
    """

    # ---- Inputs / output structure ------------------------------------------------------------
    n_obs_steps: int = 2
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # ---- Vision backbone (copied verbatim from DiffusionConfig for fairness) -----------------
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True

    # ---- Drift / diffusion network ------------------------------------------------------------
    # down_dims reused from DiffusionConfig for per-layer capacity parity with the U-Net's
    # residual blocks. Point-wise FiLM-ResNet hourglass: state → 256 → 512 → 512 → 256 → heads. 
    # (Horizon axis absent ⇒ kernel_size=1 == Linear.) DP uses (512, 1024, 2048), but we scale down for the SDE's single-step output.
    down_dims: tuple[int, ...] = (256, 512)
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    # ---- SDE specifics ------------------------------------------------------------------------
    # If sde_dt is None, defaults to 1/fps at runtime. Push-T: 0.1 s (10 Hz).
    sde_dt: float | None = 0.1
    sigma_activation: str = "exp"   # "exp" | "softplus"; used by z prior/posterior heads only

    # Train-only state-noise augmentation. >0 perturbs the drift's state window by std·√dt per frame
    # and recomputes the recon target from the perturbed anchor → corrective drift. 0.0 = legacy.
    state_noise_std: float = 0.1

    # ---- Inference -----------------------------------------------------------------------------
    # If True, drift-only inference. False → SDE noise σ_eff·√dt with σ_eff = √(kl_weight/2).
    deterministic_inference: bool = True

    # ---- Per-"episode" latent z (research_brief.md §1.2) ---------------------------------------
    # use_latent_z=False recovers the no-z PoC exactly (prior/posterior not built, no KL).
    # z_dim=8: Picked by analogy with ACT's CVAE (latent_dim=32, hidden_dim=512 → z/h = 1/16);
    # kl_weight: β on KL[q||p]. Too high → posterior collapse (q≡p, z carries no chunk info).
    #   Too low → q ignores prior (deployment z uninformed). 1e-2 .. 1.0 worth sweeping.
    # z_prior_hidden_dim / z_posterior_hidden_dim: hidden width of the (μ,σ) MLPs. None → h_dim.
    # deterministic_z_inference: use μ_p instead of sampling z at deploy. Debug/ablation only.
    # conditional_prior: True → p(z|h) (2-layer MLP, current default) and the posterior also takes
    #   h as input. False → p(z) = N(0, I) and the posterior is h-free (trajectory-only).

    use_latent_z: bool = True
    z_dim: int = 8
    z_prior_hidden_dim: int | None = None
    z_posterior_hidden_dim: int | None = None
    kl_weight: float = 1e-3          # β on KL; also effective_sigma² · 2 at inference
    kl_min: float = 0.0 # Per-dim KL floor in nats (free bits).
    z_sigma_min: float = 1e-6        # hard floor for z prior/posterior σ; init σ_p ≈ 1 (exp) or ≈ 0.69 (softplus)
    deterministic_z_inference: bool = False
    conditional_prior: bool = True

    # Train-only h conditioning-dropout: with this prob, replace h by a learnable null embedding
    # (at source, so prior/posterior/drift all see it). Inference always uses real h. 0.0 = off.
    h_dropout_prob: float = 0.1

    # z_sampling_mode: "per_chunk" → z resampled with h every n_action_steps (chunk-local posterior).
    #                  "per_episode" → z sampled once per episode (full-trajectory posterior).
    z_sampling_mode: str = "per_chunk"

    # ---- VQ-VAE variant (mutually exclusive with the Gaussian CVAE) ---------------------------
    # Discrete latent (van den Oord et al. 1711.00937): deterministic posterior + vector
    # quantizer, categorical prior p(k|h) trained with CE on the posterior's index.
    # Requires extras `lerobot[latent_sde]`.
    use_vq: bool = False
    vq_codebook_size: int = 8
    vq_commit_weight: float = 1.0 # matches VectorQuantize default
    vq_decay: float = 0.8 # matches VectorQuantize default
    vq_prior_weight: float = 1.0

    # ---- Optimization --------------------------------------------------------------------------
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Skip the last `drop_n_last_frames` anchors of each episode at sampling time so chunks
    # never extend past episode end. None → auto = `n_action_steps - 1` (no padded actions).
    drop_n_last_frames: int | None = None

    # ---- Training presets (copied verbatim from DiffusionConfig for fairness) ----------------
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.n_action_steps - 1
        if self.drop_n_last_frames < 0:
            raise ValueError(f"`drop_n_last_frames` must be >= 0. Got {self.drop_n_last_frames}.")

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if self.n_action_steps < 1:
            raise ValueError(f"`n_action_steps` must be >= 1. Got {self.n_action_steps}.")

        if self.sde_dt is not None and self.sde_dt <= 0:
            raise ValueError(f"`sde_dt` must be positive (or None). Got {self.sde_dt}.")

        if self.sigma_activation not in ("exp", "softplus"):
            raise ValueError(
                f"`sigma_activation` must be 'exp' or 'softplus'. Got {self.sigma_activation!r}."
            )
        if self.z_sigma_min <= 0:
            raise ValueError(f"`z_sigma_min` must be > 0. Got {self.z_sigma_min}.")

        if self.kl_weight < 0:
            raise ValueError(f"`kl_weight` must be non-negative. Got {self.kl_weight}.")

        if self.state_noise_std < 0:
            raise ValueError(f"`state_noise_std` must be non-negative. Got {self.state_noise_std}.")
        if not (0.0 <= self.h_dropout_prob <= 1.0):
            raise ValueError(f"`h_dropout_prob` must be in [0, 1]. Got {self.h_dropout_prob}.")

        if self.z_dim <= 0:
            raise ValueError(f"`z_dim` must be positive. Got {self.z_dim}.")
        if self.z_prior_hidden_dim is not None and self.z_prior_hidden_dim <= 0:
            raise ValueError(
                f"`z_prior_hidden_dim` must be positive or None. Got {self.z_prior_hidden_dim}."
            )
        if self.z_posterior_hidden_dim is not None and self.z_posterior_hidden_dim <= 0:
            raise ValueError(
                f"`z_posterior_hidden_dim` must be positive or None. Got "
                f"{self.z_posterior_hidden_dim}."
            )

        if self.use_vq:
            if not self.use_latent_z:
                raise ValueError("`use_vq=True` requires `use_latent_z=True`.")
            if self.vq_codebook_size < 2:
                raise ValueError(
                    f"`vq_codebook_size` must be >= 2. Got {self.vq_codebook_size}."
                )
            if not (0.0 < self.vq_decay <= 1.0):
                raise ValueError(f"`vq_decay` must be in (0, 1]. Got {self.vq_decay}.")
            if self.vq_commit_weight < 0 or self.vq_prior_weight < 0:
                raise ValueError("`vq_commit_weight` and `vq_prior_weight` must be non-negative.")

        if self.z_sampling_mode not in ("per_chunk", "per_episode"):
            raise ValueError(
                f"`z_sampling_mode` must be 'per_chunk' or 'per_episode'. Got {self.z_sampling_mode}."
            )
        if self.z_sampling_mode == "per_episode" and self.conditional_prior:
            logging.warning(
                "z_sampling_mode='per_episode' is incompatible with conditional_prior=True "
                "falling back to conditional_prior=False."
            )
            self.conditional_prior = False

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        # Image stream: past n_obs_steps frames only (vision encoder cost matches DP).
        # State stream gets a longer window via `observation_delta_indices_per_key`.
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def observation_delta_indices_per_key(self) -> dict[str, list[int]]:
        # State: past n_obs_steps + next H-1 frames, so compute_loss sees the actual demo
        # state trajectory instead of teacher-forcing from demo actions. per_episode mode
        # keeps this same narrow window — the full episode trajectory is served via an
        # in-RAM cache populated by `set_train_dataset()` at training start.
        return {OBS_STATE: list(range(1 - self.n_obs_steps, self.n_action_steps))}

    @property
    def action_delta_indices(self) -> list:
        # H = n_action_steps consecutive action targets per sample, anchored at "now"
        # (not shifted by n_obs_steps like DP). The SDE integrates forward from x_now;
        # DP's chunk overlaps the obs window because the U-Net denoises a chunk that
        # *includes* the observed timesteps. See modeling_latent_sde.compute_loss.
        return list(range(0, self.n_action_steps))

    @property
    def reward_delta_indices(self) -> None:
        return None
