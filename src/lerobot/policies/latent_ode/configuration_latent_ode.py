#!/usr/bin/env python
#
# LatentODE Policy — configuration for
# "Latent-ODE Policies for Hierarchical Robot Manipulation" (research_brief.md v7).
#
# PoC scope:
#   * per-episode latent strategy z with prior p_ψ(z|h) and posterior q_φ(z|h, x_seq, a_seq);
#     drift net reads z as FiLM cond alongside h (cond = concat([h, z])); net input is x_aug only;
#   * free-running-rollout reconstruction (drift integrated through the env's exact agent dynamics,
#     matched to the demo state path) + KL[q||p] (β-VAE). Inference is deterministic (drift-only).
#
# Deferred: controller-pushforward objective and (M, K) compliance heads (Tier 3).
#
# Fairness vs. DiffusionPolicy on Push-T: same vision backbone (DiffusionRgbEncoder),
# same FiLM-with-scale conditioning, same GroupNorm n_groups, same down_dims width ladder.
# Only difference: the chunk-horizon Conv1d collapses to point-wise Linear because the ODE
# is integrated one step at a time on the measured state x.

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("latent_ode")
@dataclass
class LatentODEConfig(PreTrainedConfig):
    """Configuration class for LatentODEPolicy.

    Defaults are tuned for Push-T (proprio + single camera) and mirror DiffusionPolicy
    so this PoC is a like-for-like replacement of the denoising U-Net.

    ODE roles (research_brief.md §1.2, §3):
        x  — measured robot state (proprio). Push-T: 2-D `observation.state` (agent_pos).
             The drift net reads **all n_obs_steps frames flattened** (augmented
             input, mirrors DP's state branch) so it has access to first-differences ≈
             velocity; the drift residual is still anchored to the most recent
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
             posterior q(z|h, x_seq, a_seq) provides chunk-level mode signal; loss =
             rollout-recon + kl_weight · KL[q||p]. z enters the drift-net input (FiLM cond
             is h only); the drift block structure is unchanged.

    `observation.environment_state` (e.g. Push-T's 16-D T-block pose) is currently ignored
    to keep the h-is-image-only contract clean. Add a concat-into-x or concat-into-h route
    if the experiment warrants it.

    Drift network output:
        mu — drift only, shape (B, action_dim). Action σ is NOT learned; inference is always
        deterministic (x_d = x + mu·dt, no diffusion noise). Training loss (see
        `LatentODEModel.compute_loss`): recon is the FREE-RUNNING ROLLOUT error — the drift's
        own predictions are integrated through the env's exact agent dynamics (`_pd_rollout`)
        and matched to the demo state path — plus (kl_weight/(H·action_dim)) · KL[q‖p] (β-VAE
        form). The generative model is a Gaussian observation N(x̂(z), σ_obs²) on each state, so
        the KL scaling is ELBO-exact and kl_weight = 2·σ_obs² (σ_obs = trajectory observation std).

    Push-T I/O (mirrors DiffusionConfig):
        - "observation.state" required.
        - At least one "observation.image*" key required.
        - "action" required. For Push-T, action == next end-effector pose target,
          making the kinematic-imitation assumption x_d ≈ x exact in form.

    New / different args vs. DiffusionConfig:
        horizon:          training PREDICTION / rollout chunk length. Per sample, h is encoded once and
                          the drift is rolled out horizon-1 steps free-running, matched to the demo state
                          path over the horizon (as DP predicts a horizon-length chunk).
        n_action_steps:   actions EXECUTED per replan at deployment (h & z refresh period). Mirrors DP:
                          predict `horizon`, execute the first `n_action_steps`, then re-encode h and
                          re-sample z. Requires 1 <= n_action_steps <= horizon.
        do_mask_loss_for_padding: mask copy-padded chunk ticks (episode ends) in the recon + posterior.
        dt:           Δt for one drift/integration step. Push-T fps=10 Hz → 0.1 s.
        sigma_activation: "exp" or "softplus"; used only by z prior/posterior σ heads.
        kl_weight:        β on KL[q||p]; scaled by 1/(horizon·action_dim) in the loss for ELBO-exact
                          balance (= 2·σ_obs²). See "Drift network output" above.

    Removed (no analog in single-step ODE):
        noise scheduler block, diffusion_step_embed_dim,
        num_inference_steps, kernel_size, clip_sample*.
    """

    # ---- Inputs / output structure ------------------------------------------------------------
    # horizon:        training PREDICTION / rollout chunk length. The drift is rolled horizon-1 steps
    #                 free-running, the posterior encodes the horizon-length demo chunk, and the recon
    #                 matches the horizon state path (DP predicts a horizon-length chunk).
    # n_action_steps: actions EXECUTED per replan at deployment (h & z refresh period). Mirrors DP:
    #                 predict `horizon`, execute the first `n_action_steps`, then re-encode h / re-sample
    #                 z. Requires 1 <= n_action_steps <= horizon. Default 16/8 = DP's Push-T train-16/act-8.
    n_obs_steps: int = 2
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

    # ---- Drift network ------------------------------------------------------------
    # down_dims reused from DiffusionConfig for per-layer capacity parity with the U-Net's
    # residual blocks. Point-wise FiLM-ResNet hourglass: state → 256 → 512 → 512 → 256 → heads. 
    # (Horizon axis absent ⇒ kernel_size=1 == Linear.) DP uses (512, 1024, 2048), but we scale down for the ODE's single-step output.
    down_dims: tuple[int, ...] = (256, 512)
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    # ---- Step-distance positional embedding ---------------------------------------------------
    # use_pe: embed each state's step-distance k (0..H-1) from its image-embedding timestep
    #   (DiffusionPolicy's DiffusionSinusoidalPosEmb) and concat it onto the FiLM cond alongside h.
    # pe_dim: width of that embedding; must be even and >= 4.
    use_pe: bool = False
    pe_dim: int = 64

    # ---- ODE specifics ------------------------------------------------------------------------
    # If dt is None, defaults to 1/fps at runtime. Push-T: 0.1 s (10 Hz).
    dt: float | None = 0.1
    sigma_activation: str = "exp"   # "exp" | "softplus"; used by z prior/posterior heads only

    # Train-only initial-state perturbation (DART-style). >0 shifts the rollout's START position by a
    # per-sample Gaussian offset ~N(0, state_noise_std²) in normalized state units (velocity preserved),
    # then rolls out deterministically and matches the CLEAN demo path — trains recovery back onto the
    # demo from an off-manifold start. Recon targets stay clean. 0.0 = no perturbation.
    state_noise_std: float = 0.03

    # ---- PD-rollout dynamics (env-specific; the rollout integrates the drift through these) -----
    # PushT's pusher is a KINEMATIC body (contact-force-free), so its dynamics is EXACTLY a PD position
    # controller (pusht.py: k_p=100, k_v=20); one tick = pd_n_substeps physics substeps of dt/pd_n_substeps.
    # Override for other position-controlled envs. Requires action_dim == state_dim.
    pd_k_p: float = 100.0
    pd_k_v: float = 20.0
    pd_n_substeps: int = 10

    # ---- Per-"chunk" latent z (research_brief.md §1.2) ---------------------------------------
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
    kl_weight: float = 1e-3          # β on KL; ELBO-exact KL scale = 2·σ_obs² (trajectory obs std)
    kl_min: float = 0.0 # Per-dim KL floor in nats (free bits).
    z_sigma_min: float = 1e-6        # hard floor for z prior/posterior σ; init σ_p ≈ 1 (exp) or ≈ 0.69 (softplus)
    deterministic_z_inference: bool = False
    conditional_prior: bool = True

    # ---- FSQ variant (mutually exclusive with the Gaussian CVAE) ------------------------------
    # Discrete latent via FSQ (finite scalar quantization, Mentzer et al. 2309.15505):
    # deterministic posterior → bounded scalar grid + straight-through rounding, categorical
    # prior p(k|h) trained with CE on the posterior's index. Requires extras `lerobot[latent_ode]`.
    use_vq: bool = False
    fsq_levels: tuple[int, ...] = (8, 5, 5)  # per-dim levels; #codes = prod(levels), z_dim = len(levels)
    fsq_prior_weight: float = 1e-5           # weight on prior-CE p(k|h); FSQ needs no commitment loss

    # ---- Optimization --------------------------------------------------------------------------
    # The rollout net is always torch.compiled (see LatentODEModel.__init__); compile_model opts into
    # ALSO compiling the inference net in place (off → eager inference, clean checkpoint keys).
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Skip the last `drop_n_last_frames` anchors of each episode at sampling time. None → auto =
    # `horizon - n_action_steps - n_obs_steps + 1`
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

        if self.dt is not None and self.dt <= 0:
            raise ValueError(f"`dt` must be positive (or None). Got {self.dt}.")

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
        if self.pd_n_substeps < 1:
            raise ValueError(f"`pd_n_substeps` must be >= 1. Got {self.pd_n_substeps}.")

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
            self.z_dim = len(self.fsq_levels)  # FSQ latent dim == number of levels
            if len(self.fsq_levels) < 1 or any(lvl < 2 for lvl in self.fsq_levels):
                raise ValueError(
                    f"`fsq_levels` must be a non-empty tuple of ints >= 2. Got {self.fsq_levels}."
                )
            if self.fsq_prior_weight < 0:
                raise ValueError(f"`fsq_prior_weight` must be non-negative. Got {self.fsq_prior_weight}.")


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
        # state trajectory over the whole prediction horizon (not teacher-forced from demo actions).
        return {OBS_STATE: list(range(1 - self.n_obs_steps, self.horizon))}

    @property
    def action_delta_indices(self) -> list:
        # `horizon` consecutive action targets per sample, anchored at "now" (deltas 0..horizon-1).
        # The ODE integrates forward from x_now; unlike DP the chunk does NOT overlap the obs window.
        # At deploy only the first `n_action_steps` are executed before h/z refresh.
        return list(range(0, self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
