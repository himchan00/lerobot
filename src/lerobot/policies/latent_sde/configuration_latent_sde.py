#!/usr/bin/env python
#
# LatentSDE Policy — configuration for
# "Latent-SDE Policies for Hierarchical Robot Manipulation" (research_brief.md v7).
#
# PoC scope:
#   * per-episode latent strategy z with prior p_ψ(z|h) and posterior q_φ(z|h, x_seq, a_seq);
#     drift/diffusion net reads z as FiLM cond alongside h (cond = concat([h, z])); net input is x_aug only;
#   * free-space Euler-Maruyama log-likelihood (research_brief.md §3.7) + KL[q||p] (β-VAE).
#
# Deferred: controller-pushforward objective and (M, K) compliance heads (Tier 3).
#
# Fairness vs. DiffusionPolicy on Push-T: same vision backbone (DiffusionRgbEncoder),
# same FiLM-with-scale conditioning, same GroupNorm n_groups, same down_dims width ladder.
# Only difference: the chunk-horizon Conv1d collapses to point-wise Linear because the SDE
# is integrated one step at a time on the measured state x.

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
             kl_weight · KL[q||p]. z conditions the drift net via FiLM alongside h
             (cond = concat([h, z])); the drift/diffusion block structure is unchanged.

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
        horizon:          SDE training chunk length H. Per sample, h/z are encoded once and the SDE
                          is unrolled H=horizon steps under teacher-forced demo states — the same
                          per-image-encode supervision budget as DP's horizon-length chunk loss.
        n_action_steps:   actions EXECUTED per replan at deployment (h & z refresh period). Mirrors
                          DP: unroll `horizon`, execute the first `n_action_steps`, then re-encode h
                          and re-sample z. Requires 1 <= n_action_steps <= horizon.
        do_mask_loss_for_padding: mask copy-padded chunk ticks (episode ends) in the recon + posterior.
        sde_dt:           Δt for one Euler-Maruyama step. Push-T fps=10 Hz → 0.1 s.
        sigma_activation: "exp" or "softplus"; used only by z prior/posterior σ heads.
        kl_weight:        β on KL[q||p], divided by (H·action_dim·dt) in the loss for
                          ELBO-exact balance: kl_weight = 2·σ_eff² exactly (σ_eff = SDE
                          diffusion coefficient). Inference SDE noise uses σ_eff = √(kl_weight/2)

    Removed (no analog in single-step SDE):
        noise scheduler block, diffusion_step_embed_dim,
        num_inference_steps, kernel_size, clip_sample*.
    """

    # ---- Inputs / output structure ------------------------------------------------------------
    # horizon:        SDE training chunk length H. The SDE is unrolled H steps per sample; the posterior
    #                 encodes the horizon-length demo chunk; the recon supervises all H ticks.
    # n_action_steps: actions EXECUTED per replan at deployment (h & z refresh period). Mirrors DP:
    #                 unroll `horizon`, execute the first `n_action_steps`, then re-encode h / re-sample
    #                 z. Requires 1 <= n_action_steps <= horizon. (DP's Push-T recipe is train-16/act-8.)
    n_obs_steps: int = 1
    horizon: int = 16
    n_action_steps: int = 8

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

    # ---- Step-distance positional embedding ---------------------------------------------------
    # use_pe: embed each state's step-distance k (0..H-1) from its image-embedding timestep
    #   (DiffusionPolicy's DiffusionSinusoidalPosEmb) and concat it onto the FiLM cond alongside h.
    # pe_dim: width of that embedding; must be even and >= 4.
    use_pe: bool = False
    pe_dim: int = 64

    # z_mode: how the latent z enters the drift net (no effect when use_latent_z=False).
    #   "cond"  — concat onto the FiLM cond alongside h (cond = [h, z]); net input = x_aug (legacy).
    #   "input" — concat onto the net input (input = [x_aug, z]); cond = h.
    z_mode: str = "cond"  # "cond" | "input"

    # ---- SDE specifics ------------------------------------------------------------------------
    # If sde_dt is None, defaults to 1/fps at runtime. Push-T: 0.1 s (10 Hz).
    sde_dt: float | None = 0.1
    sigma_activation: str = "exp"   # "exp" | "softplus"; used by z prior/posterior heads only

    # Train-only state-noise augmentation. >0 perturbs the drift's state window by std·√dt per frame
    # and recomputes the recon target from the perturbed anchor → corrective drift. 0.0 = legacy.
    state_noise_std: float = 0.3

    # state_noise_schedule: how the per-frame std varies across the chunk.
    #   "uniform" — same std·√dt on every frame (legacy).
    #   "linear"  — std ramps std·√dt/H → std·√dt over chunk ticks 0..H-1 (indices 1..H, so tick 0 gets
    #               std·√dt/H, NOT zero; peak at last tick); past obs frames (delta<0) get 0.
    #               `state_noise_std` is the PEAK (last-tick) std.
    state_noise_schedule: str = "uniform"  # "uniform" | "linear"

    # Recon target v* = (a_anchor − x̃)/dt (one-step full return to a demo action from the noised anchor
    # x̃). action_anchor picks which demo action a_anchor:
    #   "clean"   — the corresponding-index action a_k (== legacy corrective target).
    #   "nearest" — the action a_j of the nearest demo state x_j over the chunk (Behavior-Controllable
    #               autonomous field; identical to "clean" unless the state_noise_std tube is on).
    action_anchor: str = "nearest"  # "clean" | "nearest"

    # State fed to the z-posterior trajectory encoder (the action is always the clean demo action):
    #   "noisy" — the train-time perturbed state the drift reads (legacy consistency aug).
    #   "clean" — the clean demo state, so the same demo maps to the same z regardless of state-noise.
    posterior_state: str = "clean"  # "clean" | "noisy"

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

    # ---- Discrete-latent variant (mutually exclusive with the Gaussian CVAE) ------------------
    # use_vq=True swaps the Gaussian CVAE for a discrete latent: deterministic posterior → quantizer
    # → categorical prior p(k|h) trained with CE on the posterior's (detached) index. Requires
    # extras `lerobot[latent_sde]`. `quantizer` picks the flavor:
    #   "fsq" — finite scalar quantization (Mentzer et al. 2309.15505): bounded scalar grid +
    #           straight-through rounding. No learnable codebook / commitment loss / dead codes;
    #           z_dim is forced to len(fsq_levels), #codes = prod(fsq_levels).
    #   "vq"  — vector quantization (van den Oord et al. 1711.00937): learnable codebook + EMA
    #           updates + commitment loss. z_dim stays `z_dim`, #codes = vq_codebook_size.
    use_vq: bool = False
    quantizer: str = "fsq"  # "fsq" | "vq" (only used when use_vq=True)
    # -- FSQ (quantizer="fsq") --
    fsq_levels: tuple[int, ...] = (8, 5, 5)  # per-dim levels; #codes = prod(levels), z_dim = len(levels)
    fsq_prior_weight: float = 1e-3           # weight on prior-CE p(k|h); FSQ needs no commitment loss
    # -- VQ (quantizer="vq") --
    vq_codebook_size: int = 8                # #codes; keep <= batch_size so kmeans_init seeds every code
    vq_commit_weight: float = 1.0          # weight on commitment loss (matches VectorQuantize default)
    vq_decay: float = 0.99                    # codebook EMA decay (matches VectorQuantize default)
    vq_prior_weight: float = 1e-3            # weight on prior-CE p(k|h)

    # ---- Optimization --------------------------------------------------------------------------
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation: mask copy-padded chunk ticks (episode ends) out of the recon + posterior.
    do_mask_loss_for_padding: bool = False

    # Skip the last `drop_n_last_frames` anchors of each episode at sampling time. None → auto =
    # `max(0, horizon - n_action_steps - n_obs_steps + 1)` (DP formula). For horizon >= 2*n_action_steps
    # + n_obs_steps - 2 (the default 64/32/2 sits on this threshold) it keeps the EXECUTED region unpadded
    # and copy-pads only the predicted tail; below that some executed ticks may pad too (masked iff
    # do_mask_loss_for_padding).
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
            self.drop_n_last_frames = max(0, self.horizon - self.n_action_steps - self.n_obs_steps + 1)
        if self.drop_n_last_frames < 0:
            raise ValueError(f"`drop_n_last_frames` must be >= 0. Got {self.drop_n_last_frames}.")

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        if self.horizon < 1:
            raise ValueError(f"`horizon` must be >= 1. Got {self.horizon}.")
        if not (1 <= self.n_action_steps <= self.horizon):
            raise ValueError(
                f"`n_action_steps` must satisfy 1 <= n_action_steps <= horizon. "
                f"Got n_action_steps={self.n_action_steps}, horizon={self.horizon}."
            )

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
        if self.state_noise_schedule not in ("uniform", "linear"):
            raise ValueError(
                f"`state_noise_schedule` must be 'uniform' or 'linear'. Got {self.state_noise_schedule!r}."
            )

        if self.action_anchor not in ("clean", "nearest"):
            raise ValueError(f"`action_anchor` must be 'clean' or 'nearest'. Got {self.action_anchor!r}.")
        if self.posterior_state not in ("clean", "noisy"):
            raise ValueError(f"`posterior_state` must be 'clean' or 'noisy'. Got {self.posterior_state!r}.")
        if self.z_mode not in ("cond", "input"):
            raise ValueError(f"`z_mode` must be 'cond' or 'input'. Got {self.z_mode!r}.")

        if self.use_pe and (self.pe_dim < 4 or self.pe_dim % 2 != 0):
            raise ValueError(f"`pe_dim` must be an even int >= 4 when `use_pe=True`. Got {self.pe_dim}.")

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
            if self.quantizer not in ("fsq", "vq"):
                raise ValueError(f"`quantizer` must be 'fsq' or 'vq'. Got {self.quantizer!r}.")
            if self.quantizer == "fsq":
                self.z_dim = len(self.fsq_levels)  # FSQ latent dim == number of levels
                if len(self.fsq_levels) < 1 or any(lvl < 2 for lvl in self.fsq_levels):
                    raise ValueError(
                        f"`fsq_levels` must be a non-empty tuple of ints >= 2. Got {self.fsq_levels}."
                    )
                if self.fsq_prior_weight < 0:
                    raise ValueError(f"`fsq_prior_weight` must be non-negative. Got {self.fsq_prior_weight}.")
            else:  # "vq" — z_dim is the (configured) code dim; codebook is learnable.
                if self.vq_codebook_size < 2:
                    raise ValueError(f"`vq_codebook_size` must be >= 2. Got {self.vq_codebook_size}.")
                if not (0.0 < self.vq_decay <= 1.0):
                    raise ValueError(f"`vq_decay` must be in (0, 1]. Got {self.vq_decay}.")
                if self.vq_commit_weight < 0 or self.vq_prior_weight < 0:
                    raise ValueError("`vq_commit_weight` and `vq_prior_weight` must be non-negative.")


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
        # State: past n_obs_steps + next horizon-1 frames, so compute_loss sees the actual demo
        # state trajectory over the whole prediction horizon (not teacher-forcing from demo actions).
        return {OBS_STATE: list(range(1 - self.n_obs_steps, self.horizon))}

    @property
    def action_delta_indices(self) -> list:
        # `horizon` consecutive action targets per sample, anchored at "now" (deltas 0..horizon-1,
        # not shifted by n_obs_steps like DP). The SDE integrates forward from x_now. At deploy only
        # the first `n_action_steps` are executed before the h/z refresh.
        return list(range(0, self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
