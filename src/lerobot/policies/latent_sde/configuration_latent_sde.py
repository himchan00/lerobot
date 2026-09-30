#!/usr/bin/env python
#
# LatentSDE Policy — configuration for
# "Latent-SDE Policies for Hierarchical Robot Manipulation" (research_brief.md v7).
#
# PoC scope:
#   * per-episode latent strategy z with prior p_ψ(z|h) and posterior q_φ(z|a_seq);
#     drift/diffusion net reads z as FiLM cond alongside h (cond = concat([h, z])); net input is x_aug only;
#   * free-space Euler-Maruyama log-likelihood (research_brief.md §3.7) + KL[q||p] (β-VAE).
#
# Deferred: controller-pushforward objective and (M, K) compliance heads (Tier 3).
#
# Fairness vs. DiffusionPolicy on Push-T: same vision backbone (DiffusionRgbEncoder),
# same FiLM-with-scale conditioning, same GroupNorm n_groups, same down_dims width ladder.
# Only difference: the chunk-horizon Conv1d collapses to point-wise Linear because the SDE
# is integrated one step at a time on the measured state x.

import copy
from dataclasses import dataclass

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.configs.parser import parse_arg
from lerobot.optim import AdamConfig, CosineDecayWithWarmupSchedulerConfig, DiffuserSchedulerConfig
from lerobot.utils.constants import OBS_STATE

# Env-dependent defaults: fields left None are filled in __post_init__ from ENV_PRESETS[--env.type];
# explicit values win. pusht = the DP recipe, libero = the SmolVLA recipe.
ENV_PRESETS = {
    "pusht": {
        "context_encoder": "resnet",
        "conditioning": "film",
        "sde_geometry": "euclidean",
        "n_obs_steps": 2,
        "horizon": 16,
        "n_action_steps": 8,
        "crop_shape": (84, 84),
        "normalization_mapping": {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        "do_mask_loss_for_padding": False,
        "optimizer_lr": 1e-3,
        "optimizer_betas": (0.95, 0.999),
        "optimizer_weight_decay": 1e-6,
        "scheduler_name": "cosine",
        "scheduler_warmup_steps": 500,
    },
    "libero": {
        "context_encoder": "smolvlm2",
        "conditioning": "token_kv",
        "sde_geometry": "so3_r3_body",
        "n_obs_steps": 1,
        "horizon": 50,
        "n_action_steps": 10,
        "crop_shape": None,
        # Body geometry reads raw poses and controller commands.
        "normalization_mapping": {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        "do_mask_loss_for_padding": True,
        # SmolVLA's values; it uses AdamW, identical to Adam at this weight decay.
        "optimizer_lr": 1e-4,
        "optimizer_betas": (0.9, 0.95),
        "optimizer_weight_decay": 1e-10,
        "scheduler_name": "cosine_decay_with_warmup",
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 30_000,
        "scheduler_decay_lr": 2.5e-6,
    },
}


@PreTrainedConfig.register_subclass("latent_sde")
@dataclass
class LatentSDEConfig(PreTrainedConfig):
    """Configuration class for LatentSDEPolicy.

    Env-dependent fields default to None and are filled from ENV_PRESETS by `--env.type`. The Push-T
    preset mirrors DiffusionPolicy so this PoC is a like-for-like replacement of the denoising U-Net.

    SDE roles (research_brief.md §1.2, §3):
        x  — measured robot state (proprio). Push-T: 2-D `observation.state` (agent_pos).
             The drift/diffusion net reads the current frame on every Tier-2 tick. In Euclidean
             mode, the one-step residual is anchored to that frame (`mean = x_t + s·μ`).
             At training time the trailing horizon states are sampled directly from the dataset
             (see `observation_delta_indices_per_key`), matching the deployment-time stream.
        h  — joint observation conditioning (Tier-1). ResNet image features are concatenated
             with an embedding of the causal state window. SmolVLM2 receives image-language
             tokens plus one projected token per causal state frame, ordered oldest to newest.
             Refreshed every `n_action_steps` ticks so the context-encoder duty cycle
             matches DiffusionPolicy's (fair compute) and the Tier-1/Tier-2 rate split
             is reproduced architecturally. cf. notes/h_is_conditioning.tex
        z  — per-episode latent strategy. CVAE-style: prior p(z|h) is re-sampled at
             deployment **in lock-step with every h refresh** ("episode" =
             one h-refresh window), committing each chunk to one mode. At training,
             posterior q(z|a_seq) provides chunk-level mode signal; loss = NLL +
             beta · KL[q||p]. z conditions the drift net via FiLM alongside h
             (cond = concat([h, z])); the drift/diffusion block structure is unchanged.

    Euclidean drift/diffusion network output:
        mu — the standardized one-step action, shape (B, action_dim), in units of the action scale s
        (see action_scale). Inference returns the target state x + s·(mu + σ·ε).
        There is no model-time Δt: one step per action, at the environment's own rate.
        Training loss (Gaussian path): nll + beta·KL[q‖p], nll the Gaussian NLL of d*/s ~ N(μ, σ²)
        over the H·action_dim steps (d* = a−x), nll and KL both
        /(H·action_dim). σ² is NOT gradient-trained — an EMA of the per-batch MLE mean‖d*/s−μ‖²
        (σ-VAE); beta is the β-VAE coefficient on KL. The optimized loss is (nll + beta·KL)·sg(2σ²),
        so the recon gradient equals the plain MSE gradient and grad norms match the VQ/FSQ path
        (logged nll_loss / kl_loss stay unscaled). (VQ/FSQ keep an MSE recon + commitment/prior-CE.)

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
        sde_geometry:     "euclidean" predicts a target-state residual (action_dim == state_dim);
                          "so3_r3_body" predicts body-local increments of relative pose commands plus
                          the gripper (same scalar variance EMA).
        action_scale:     per-dimension std of the one-step action target, swept from the dataset.
        sigma_activation: "exp" or "softplus"; used only by z prior/posterior σ heads.
        beta:             β-VAE coefficient on KL[q||p] in the ELBO loss = nll + beta·KL. The
                          action-decoder σ² (one scalar in both modes) is calibrated by EMA to the
                          analytic MLE (σ-VAE), not gradient-trained (see sigma_ema_decay).

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
    # None → ENV_PRESETS (also every other `| None = None` field listed there).
    n_obs_steps: int | None = None
    horizon: int | None = None
    n_action_steps: int | None = None

    normalization_mapping: dict[str, NormalizationMode] | None = None

    # ---- Context / action representation capabilities ----------------------------------------
    context_encoder: str | None = None  # "resnet" | "smolvlm2"
    conditioning: str | None = None  # "film" | "token_kv"; token_kv requires SmolVLM2

    sde_geometry: str | None = None  # "euclidean" | "so3_r3_body"

    # The generic SmolVLM2 backbone (Hub main, same as SmolVLA's default) is kept frozen.
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    vlm_num_layers: int = 16
    vlm_context_dim: int = 512
    vlm_resize_shape: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 48

    # ---- Vision backbone (copied verbatim from DiffusionConfig for fairness) -----------------
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 64
    use_separate_rgb_encoder_per_camera: bool = True

    # ---- Drift / diffusion network ------------------------------------------------------------
    # down_dims reused from DiffusionConfig for per-layer capacity parity with the U-Net's
    # residual blocks. Point-wise FiLM-ResNet hourglass: state → 256 → 512 → 512 → 256 → heads. 
    # (Horizon axis absent ⇒ kernel_size=1 == Linear.) DP uses (512, 1024, 2048), but we scale down for the SDE's single-step output.
    down_dims: tuple[int, ...] = (512, 1024)
    n_groups: int = 8
    use_film_scale_modulation: bool = True

    # z_mode: where the latent z enters the drift net (no effect when use_latent_z=False):
    #   "cond"  — append z to FiLM conditioning, or add its projection to the token-KV query.
    #   "input" — concat z with the current state before the drift's input projection.
    z_mode: str = "cond"  # "cond" | "input"

    # drift_uses_h: provide joint context directly to the drift (FiLM vector or prefix K/V).
    #   False removes that direct route; deployment context can still reach the field through
    #   z sampled from p(z|h). The current state remains the drift input in either mode.
    #   Acts independently of z_mode; without z this is a proprio-only ablation.
    drift_uses_h: bool = True

    # ---- SDE specifics ------------------------------------------------------------------------
    # One drift step per action at the environment's own rate; there is no separate model-time Δt.
    sigma_activation: str = "exp"   # "exp" | "softplus"; used by z prior/posterior heads only

    # action_scale: per-dim std s of the one-step target d (a − x / body increment); the drift predicts
    #   d/s. None → swept from the dataset before training and cached (see action_scale.py).
    action_scale: list[float] | None = None

    # Action-decoder σ² (SDE diffusion coeff²) is NOT gradient-trained: it's EMA'd toward the analytic
    # per-batch MLE mean‖d*/s−μ‖² (calibrated σ-VAE, arXiv:2006.13202). sigma_ema_decay = EMA decay.
    # One scalar in every mode (so3_r3_body included): with the loss rescale sg(2σ²) the recon
    # gradient is then exactly the unweighted MSE gradient over all action coordinates.
    sigma_ema_decay: float = 0.99

    # Train-only state-noise augmentation: perturbs the drift's state window by std·s per frame (a fraction
    # of a typical one-step action) and recomputes the corrective target. 0.0 = off.
    state_noise_std: float = 0.3

    # state_noise_schedule: how the per-frame std varies across the chunk.
    #   "uniform" — same std·s on every frame.
    #   "linear"  — std ramps std·s/H → std·s over chunk ticks 0..H-1 (indices 1..H, so tick 0 gets
    #               std·s/H, NOT zero; peak at last tick); past obs frames (delta<0) get 0.
    #               `state_noise_std` is the PEAK (last-tick) std.
    state_noise_schedule: str = "uniform"  # "uniform" | "linear"

    # Recon target d* = a_anchor − x̃ (one-step full return to a demo action from the noised anchor
    # x̃). action_anchor picks which demo action a_anchor:
    #   "clean"   — the corresponding-index action a_k (== legacy corrective target).
    #   "nearest" — the action a_j of the nearest demo state x_j over the chunk (Behavior-Controllable
    #               autonomous field; identical to "clean" unless the state_noise_std tube is on).
    action_anchor: str = "nearest"  # "clean" | "nearest"

    # normalize_state: express DRIFT inputs relative to the clean chunk-initial state x_0.
    #   Euclidean inputs subtract x_0. Pose inputs use R_0.T @ (p - p_0) and R_0.T @ R, keeping
    #   the continuous rotation-column features and absolute finger positions. The origin is fixed
    #   until the next h/z refresh. Joint observation context stays absolute.
    #   Euclidean POSTERIOR actions subtract x_0; so3_r3_body's posterior reads the clean body-frame
    #   increment / s.
    #   Drift targets, integration, and controller conversion keep their original coordinates.
    normalize_state: bool = True

    # ---- Inference -----------------------------------------------------------------------------
    # If True, drift-only inference. False → SDE noise s·σ·ε with σ from the action_var EMA.
    deterministic_inference: bool = True

    # ---- Per-"episode" latent z (research_brief.md §1.2) ---------------------------------------
    # use_latent_z=False recovers the no-z PoC exactly (prior/posterior not built, no KL).
    # z_dim=8: Picked by analogy with ACT's CVAE (latent_dim=32, hidden_dim=512 → z/h = 1/16);
    # beta: β-VAE coefficient on KL[q||p]. Too high → posterior collapse (q≡p, z carries no chunk info).
    #   Too low → q ignores prior (deployment z uninformed). 1e-2 .. 1.0 worth sweeping.
    # z_prior_hidden_dim / z_posterior_hidden_dim: hidden width of the (μ,σ) MLPs. None uses the
    # original visual-window width for ResNet, or vlm_context_dim for SmolVLM2.
    # deterministic_z_inference: use μ_p instead of sampling z at deploy. Debug/ablation only.

    use_latent_z: bool = True
    z_dim: int = 8
    z_prior_hidden_dim: int | None = None
    z_posterior_hidden_dim: int | None = None
    beta: float = 1.0                # β-VAE coefficient on KL[q‖p] in the ELBO (loss = nll + beta·KL)
    z_sigma_min: float = 1e-6        # hard floor for z prior/posterior σ; init σ_p ≈ 1 (exp) or ≈ 0.69 (softplus)
    deterministic_z_inference: bool = False

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
    do_mask_loss_for_padding: bool | None = None

    # Skip the last `drop_n_last_frames` anchors of each episode at sampling time. None → 0 with
    # do_mask_loss_for_padding (padding is masked), else the DP formula
    # `max(0, horizon - n_action_steps - n_obs_steps + 1)`. For horizon >= 2*n_action_steps
    # + n_obs_steps - 2 (the default 64/32/2 sits on this threshold) it keeps the EXECUTED region unpadded
    # and copy-pads only the predicted tail; below that some executed ticks may pad too (masked iff
    # do_mask_loss_for_padding).
    drop_n_last_frames: int | None = None

    # ---- Training presets (ENV_PRESETS: pusht = DiffusionConfig's, libero = SmolVLAConfig's) ----
    optimizer_lr: float | None = None
    optimizer_betas: tuple | None = None
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float | None = None
    scheduler_name: str | None = None  # diffusers name (e.g. "cosine") | "cosine_decay_with_warmup"
    scheduler_warmup_steps: int | None = None
    scheduler_decay_steps: int | None = None  # cosine_decay_with_warmup only
    scheduler_decay_lr: float | None = None  # cosine_decay_with_warmup only

    def __post_init__(self):
        super().__post_init__()

        unset = [key for key in ENV_PRESETS["pusht"] if key != "crop_shape" and getattr(self, key) is None]
        if unset:
            env_type = parse_arg("env.type")  # the training CLI's env; saved configs are already complete
            if env_type not in ENV_PRESETS:
                raise ValueError(
                    f"latent_sde: {unset} default per env; set --env.type to one of {list(ENV_PRESETS)} "
                    "or pass them explicitly."
                )
            for key, value in ENV_PRESETS[env_type].items():
                if getattr(self, key) is None:
                    setattr(self, key, copy.deepcopy(value))

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = (
                0
                if self.do_mask_loss_for_padding
                else max(0, self.horizon - self.n_action_steps - self.n_obs_steps + 1)
            )

        if not (1 <= self.n_action_steps <= self.horizon):
            raise ValueError(
                f"`n_action_steps` must satisfy 1 <= n_action_steps <= horizon. "
                f"Got n_action_steps={self.n_action_steps}, horizon={self.horizon}."
            )

        if self.state_noise_schedule not in ("uniform", "linear"):
            raise ValueError(
                f"`state_noise_schedule` must be 'uniform' or 'linear'. Got {self.state_noise_schedule!r}."
            )
        if self.action_anchor not in ("clean", "nearest"):
            raise ValueError(f"`action_anchor` must be 'clean' or 'nearest'. Got {self.action_anchor!r}.")

        if self.z_mode not in ("cond", "input"):
            raise ValueError(f"`z_mode` must be 'cond' or 'input'. Got {self.z_mode!r}.")

        if self.use_vq:
            if not self.use_latent_z:
                raise ValueError("`use_vq=True` requires `use_latent_z=True`.")
            if self.quantizer not in ("fsq", "vq"):
                raise ValueError(f"`quantizer` must be 'fsq' or 'vq'. Got {self.quantizer!r}.")
            if self.quantizer == "fsq":
                self.z_dim = len(self.fsq_levels)  # FSQ latent dim == number of levels

        if self.context_encoder == "resnet" and self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig | CosineDecayWithWarmupSchedulerConfig:
        if self.scheduler_name == "cosine_decay_with_warmup":
            return CosineDecayWithWarmupSchedulerConfig(
                peak_lr=self.optimizer_lr,
                decay_lr=self.scheduler_decay_lr,
                num_warmup_steps=self.scheduler_warmup_steps,
                num_decay_steps=self.scheduler_decay_steps,
            )
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        """Required config interface; callers provide the documented feature layout."""

    @property
    def observation_delta_indices(self) -> list:
        # Image stream: past n_obs_steps frames only (context encoder cost matches DP).
        # State stream gets a longer window via `observation_delta_indices_per_key`.
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def observation_delta_indices_per_key(self) -> dict[str, list[int]]:
        # State always includes the causal n_obs_steps context window followed by the horizon-length
        # teacher-forced trajectory. The current frame (delta 0) belongs to both slices.
        return {OBS_STATE: list(range(1 - self.n_obs_steps, self.horizon))}

    @property
    def action_delta_indices(self) -> list:
        # `horizon` consecutive action targets per sample, anchored at "now" (deltas 0..horizon-1,
        # not shifted by n_obs_steps like DP). Euclidean target states integrate from x_now; body
        # commands from zero. At deploy only the first `n_action_steps` are executed before
        # the h/z refresh.
        return list(range(0, self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
