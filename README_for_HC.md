# HC 개인용 메모

## 1. 환경 세팅 (연구실 GPU 서버, Ubuntu 22.04 — robot9에서 확인)

레포 루트(robot9: `/PublicSSD/himchan/lerobot`)에서.

### Push-T

```bash
# python과 ffmpeg를 conda-forge 한 채널에서 같이 설치 (defaults 채널과 섞으면 torchcodec이 FFmpeg를 못 불러옴)
conda create -y -n latent_sde --override-channels -c conda-forge python=3.12 "ffmpeg=7"
conda activate latent_sde
pip install -e '.[pusht,diffusion,training,latent_sde]'

# Ubuntu 22.04: env의 libstdc++를 먼저 로드 (torchcodec의 CXXABI_1.3.15 에러 방지)
conda env config vars set LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
# wandb 로그인(서버당 한 번) + entity 고정 (계정 기본 entity piggene00은 권한 없음)
wandb login
conda env config vars set WANDB_ENTITY=himchan00
conda deactivate && conda activate latent_sde
```

### LIBERO (위 env에 추가)

```bash
# egl_probe는 pip 격리 빌드에서 cmake를 못 찾음 → 격리 없이 먼저 빌드
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install --no-build-isolation egl_probe==1.0.2 hf-egl-probe==1.0.2
pip install -e '.[pusht,diffusion,training,latent_sde,libero,smolvla]'
# LIBERO asset (경로 질문에는 N)
echo N | python -c 'from libero.libero.utils.download_utils import download_assets_from_huggingface; download_assets_from_huggingface()'

# 학습 전마다: 캐시는 PublicSSD (robot9엔 PublicHDD 없음), 헤드리스 렌더링
export HF_HOME=/PublicSSD/himchan/hf_cache MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
```

- tmux 서버를 `latent_sde`가 켜진 셸에서 띄웠으면 새 창에서 `conda activate`가 무시되어 `LD_PRELOAD`/`WANDB_ENTITY`가 안 잡힘 → `conda deactivate` 후 activate.
- MuJoCo 종료 때 나오는 `Exception ignored in ... EGLError`는 무시해도 됨.

## 2. 학습 커맨드

### 2.1 Push-T + Diffusion Policy

```bash
lerobot-train \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --policy.crop_shape=[84,84] \
    --policy.crop_is_random=true \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    --dataset.repo_id=lerobot/pusht \
    --env.type=pusht \
    --output_dir=outputs/train/diffusion_pusht \
    --job_name=diffusion_pusht \
    --batch_size=64 \
    --seed=0 \
    --eval.use_async_envs=false \
    --wandb.enable=True \
    --wandb.project=lerobot_pusht
```

- 96×96 이미지를 학습 때 84×84 랜덤 크롭 (원 DP 논문 설정). 이 포크는 `crop_shape` 기본값이 `None`이라 명시.
- `--eval.use_async_envs=false`: async 워커에서 gym_pusht가 등록되지 않아 생기는 NamespaceNotFound 우회.

### 2.2 Push-T + Latent-SDE (`latent_sde_z16_2_16_8_no_h_zeroed`, 50%)

```bash
lerobot-train \
    --policy.type=latent_sde \
    --env.type=pusht \
    --dataset.repo_id=lerobot/pusht \
    --policy.z_dim=16 \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/latent_sde_z16_2_16_8_no_h_zeroed \
    --job_name=latent_sde_z16_2_16_8_no_h_zeroed \
    --batch_size=64 \
    --seed=0 \
    --steps=200000 \
    --eval.use_async_envs=false \
    --eval.batch_size=50 \
    --wandb.enable=true \
    --wandb.project=lerobot_pusht
```

- `--env.type=pusht` 프리셋: ResNet, `n_obs_steps` 2 / `horizon` 16 / `n_action_steps` 8, crop 84, DP 정규화, Adam 1e-3. drift는 z만 FiLM으로 받음(관측은 ResNet h → MLP prior → z). 그 외 기본값: Gaussian z, β=1, `normalize_state=true`.
- 50% = 160k/180k/200k 학습 중 eval(각 50 에피소드) 평균 (wandb `hnw2d31e`, 최고 54%). 이 run은 9/30 코드(state noise 0.1·√dt, FiLM 생성기 Mish→Linear)로 학습됨. 그 noise는 지금 단위로 0.378이고, 위 명령은 preset 0.3을 씀. FiLM 생성기는 지금 Linear(z).
- 참고: drift가 h도 받던 이전 구조(`latent_sde_z16_2_16_8_h_init_state_zeroed`)는 63% (160k/180k/200k × 100 에피소드).

### 2.3 LIBERO + SmolVLA

```bash
lerobot-train \
    --policy.type=smolvla \
    --policy.load_vlm_weights=true \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --dataset.repo_id=lerobot/libero \
    --dataset.video_backend=torchcodec \
    --steps=30000 \
    --batch_size=64 \
    --save_freq=5000 \
    --env_eval_freq=10000 \
    --eval.n_episodes=1 \
    --eval.use_async_envs=false \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/smolvla_libero_30k \
    --job_name=smolvla_libero_30k \
    --wandb.enable=true \
    --wandb.project=lerobot_libero
```

- SmolVLA 자체 프리셋(AdamW 1e-4, warmup 1,000, cosine 30k → 2.5e-6)이 3절 세팅과 같아서 optimizer 인자는 필요 없음. `n_action_steps=10`은 평가 때 재계획 주기에만 영향.
- VLM: SmolVLA와 Latent-SDE 모두 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`의 Hub `main`을 불러옴 (현재 rev `7b375e1b`).

### 2.4 LIBERO + Latent-SDE (SO3)

```bash
lerobot-train \
    --policy.type=latent_sde \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --dataset.repo_id=lerobot/libero \
    --dataset.video_backend=torchcodec \
    --steps=30000 \
    --batch_size=64 \
    --save_freq=5000 \
    --env_eval_freq=10000 \
    --eval.n_episodes=1 \
    --eval.use_async_envs=false \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/latent_sde_so3_libero_30k \
    --job_name=latent_sde_so3_libero_30k \
    --wandb.enable=true \
    --wandb.project=lerobot_libero
```

- 나머지 설정은 전부 `--env.type=libero` 프리셋 (3절). z16 변형: `--policy.z_dim=16`.
- robot9: `outputs/queue_libero/queue8.sh` (tmux `libero_queue`)로 `latent_sde_so3_libero_30k` → `..._30k_z16` → `smolvla_libero_30k` 순서로 실행 (run당 약 3.5h). 5090, batch 64에서 메모리 약 14GB, 0.376 s/step.

## 3. LIBERO 학습 세팅과 근거 (논문용)

Latent-SDE의 LIBERO 설정은 `--env.type=libero` 프리셋(`ENV_PRESETS`, `configuration_latent_sde.py`)에 들어 있고, 학습 길이만 CLI로 준다.
기준은 SmolVLA(같은 SmolVLM2 백본을 쓰는 VLA 베이스라인)의 LIBERO 레시피이며, 바꾼 곳은 아래 "SmolVLA 논문과 다른 점"에 근거와 함께 정리했다.

### 데이터와 관측
| 항목 | 값 | 근거 |
|---|---|---|
| 데이터 | `lerobot/libero` (lerobot 기본 `v3.0` 태그 = 현재 rev `a1aaacb7`), 1,693 에피소드, 40 태스크, 273,465 프레임, 10 fps | SmolVLA 논문 §4.1의 LIBERO 데이터(`physical-intelligence/libero`, 1,693 에피소드)와 같은 에피소드 수. `lerobot/smolvla_libero` 체크포인트도 이 데이터로 학습 |
| 관측 | agentview + wrist 256×256, SmolVLM 입력 512×512 | 논문 §4.3 "images resized to 512×512" (`vlm_resize_shape`) |
| state / action | state 8D `[p, axis-angle, finger qpos]`, action 7D 상대 OSC_POSE 명령 + 그리퍼 ±1 | 데이터셋 그대로 (robosuite OSC_POSE: 명령 1 = 0.05 m / 0.5 rad) |

### 모델
| 항목 | 값 | 근거 |
|---|---|---|
| 문맥 인코더 | SmolVLM2-500M-Video-Instruct (Hub `main`, 현재 rev `7b375e1b`) 고정, 앞 16층 | 논문 §4.3: VLM 고정, LLM 앞 16층만 사용, 주 모델(0.45B)은 SmolVLM2-500M. SmolVLA 베이스라인과 같은 모델 |
| prior / drift | prior: 학습되는 query 1개가 층별 VLM K/V를 읽는 token-KV expert. drift: z만 FiLM으로 받는 가벼운 MLP (512→1024→1024→512) | SmolVLA action expert의 토큰 접근 방식을 prior에 씀 (논문 Table 6: CA가 SA보다 좋음). drift는 매 tick 도는 빠른 네트워크 |
| 액션 기하 | `so3_r3_body`: 현재 EEF body frame 증분, 정규화 IDENTITY | LIBERO 액션이 상대 OSC 명령이라 물리 단위(pose)에서 증분을 정의해야 함. 표준화는 `action_scale`로 따로 함 |
| 관측 창 / 학습 구간 | `n_obs_steps=1`, `horizon=50` | SmolVLA 기본값(`n_obs_steps=1`, `chunk_size=50`), 논문 Table 12: chunk 10~50이 좋음 |
| 재계획 주기 | `n_action_steps=10` (h·z를 10 tick마다 새로 계산; drift는 매 tick 현재 state로 동작) | 논문 Table 13 (chunk 50 고정): 실행 스텝 1 → 80.3, **10 → 82.8**, 30 → 70.8, 50 → 51.8 |
| 에피소드 끝 처리 | `drop_n_last_frames=0` + padding 마스크 | SmolVLA도 padding action을 loss에서 가리고 프레임을 버리지 않음. DP 공식(50−10−1+1=40)이면 에피소드(평균 162프레임) 끝 40개 = 약 25%, 과제를 마무리하는 구간이 빠짐 |
| 액션 표준화 | `action_scale`: 데이터셋 전체의 한 스텝 목표 표준편차 (위치 0.390, 회전 0.062, 그리퍼 1.0 명령 단위) | drift 목표가 차원마다 분산 약 1이 되도록 |
| posterior 입력 | body frame 증분 / s (= noise 없는 drift 목표) | drift 목표와 같은 좌표·스케일 |
| state noise | 없음 (`state_noise_std=0`, preset) | 0.05 / 0.1 / 0.2를 시험했지만 성능 향상 없음 (Push-T preset은 0.3) |
| latent z | Gaussian CVAE, `z_dim=8`, β=1 (z16 변형 비교) | 기본값 |

### 최적화
| 항목 | 값 | 근거 |
|---|---|---|
| optimizer | Adam, lr 1e-4, β=(0.9, 0.95), eps 1e-8, weight decay 1e-10, grad clip 10 | 논문 §4.3 (AdamW, β₁=0.9, β₂=0.95, lr 1e-4) + lerobot `SmolVLAConfig` (eps, wd, clip). SmolVLA는 AdamW지만 wd 1e-10에서는 Adam과 같음: AdamW의 감쇠 θ(1−lr·wd)=θ(1−1e-14)는 fp32에서 그대로 θ이고, Adam의 L2 항 1e-10·θ는 대부분 gradient의 fp32 분해능보다 작음 |
| 학습률 스케줄 | warmup 1,000, cosine으로 1e-4 → 2.5e-6, decay 30k = 전체 학습 길이 | 논문 §4.3 (cosine, 최소 2.5e-6) + `SmolVLAConfig` (warmup 1,000, decay 30k). 이 값은 openpi `CosineDecaySchedule` 기본값이고, openpi는 `pi0_libero`를 정확히 30k 스텝 학습함 (decay = 학습 길이) |
| 학습 길이 | 30,000 스텝 × batch 64 (약 192만 샘플, 약 7 epoch) | 논문의 시뮬레이션 fine-tune은 100k 스텝 / batch 64. 하지만 HF 공식 SmolVLA LIBERO 체크포인트(`HuggingFaceVLA/smolvla_libero_ckpts`, `train_config.json`)도 decay 30k로 100k를 학습해 30k 이후 70%는 최소 학습률 2.5e-6에 고정됨. 그래서 스케줄이 실제로 쓰는 30k에서 멈춤 (openpi `pi0_libero`와 같은 길이). 계산 비용 약 1/3 |
| batch | 64 | 논문 §4.3 시뮬레이션 fine-tune (HF 공식 체크포인트는 32) |
| 정밀도 | fp32 학습 (AMP 없음), 고정 VLM은 bf16 로드 | 논문은 bf16 + `torch.compile` (속도용). 이 모델은 VLM이 이미 bf16이고 학습 부분이 작아 bf16 autocast가 오히려 약 7% 느림 (5090: 0.402 vs 0.376 s/step, loss 동일) |

### 평가
- 학습 중 모니터링: 10k마다, 4 suite × 10 태스크 × 태스크당 1회 (40 에피소드), `n_action_steps=10`. 잡음이 커서 추세 확인용.
- 논문 수치: 논문 §4.1과 같게 **태스크당 10회** (suite당 100, 총 400 에피소드)로 최종 체크포인트를 따로 평가하고 suite별 + 평균 성공률을 보고할 것 (TODO).
- SmolVLA 베이스라인: 2.3 명령으로 같은 데이터·batch·30k 스텝 학습, 같은 `n_action_steps=10`으로 평가 (TODO). 논문 프로토콜(`n_action_steps=1`) 결과는 참고용으로 같이 보고.

### SmolVLA 논문과 다른 점 (논문에 쓸 것)
- 학습 30k 스텝 (논문 100k): 기준 스케줄이 30k 이후 최소 학습률이라 추가 70k의 효과가 작고, 계산 비용을 1/3로 줄임. 베이스라인도 같은 30k로 학습해 공정성 유지.
- 재계획 10스텝 (논문 시뮬레이션은 매 스텝): 논문 Table 13에서 10이 가장 좋고, latent_sde는 drift가 매 tick state 피드백을 받아 재계획 주기가 h·z 갱신에만 해당.
- Adam (SmolVLA는 AdamW): wd 1e-10에서 수치적으로 같음.
- fp32 학습 (논문 bf16): 속도 이득이 없어서 (위 정밀도 행).

### 버전 고정 방침
- 데이터와 VLM 모두 고정하지 않음: 데이터는 lerobot 기본 `v3.0` 태그, VLM은 Hub `main` (SmolVLA와 Latent-SDE가 항상 같은 것을 받음).
- 논문용 기록 (2026-10-01 학습 기준): `lerobot/libero` rev `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`, `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` rev `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

### 참고 출처
- SmolVLA 논문: arXiv 2506.01844 — §4.1 (LIBERO 데이터·평가), §4.3 (구현 세부), Table 12 (chunk 크기), Table 13 (실행 스텝).
- lerobot `src/lerobot/policies/smolvla/configuration_smolvla.py`: optimizer/scheduler 프리셋 (lr 1e-4, β (0.9, 0.95), eps 1e-8, wd 1e-10, clip 10, warmup 1,000, decay 30k → 2.5e-6).
- HF 공식 체크포인트 `HuggingFaceVLA/smolvla_libero_ckpts` (`100000/pretrained_model/train_config.json`): 100k 스텝, batch 32, decay 30k, `n_action_steps=1`, `physical-intelligence/libero`.
- openpi `src/openpi/training/optimizer.py` (`CosineDecaySchedule`: warmup 1,000, peak 2.5e-5, decay 30k, 2.5e-6), `src/openpi/training/config.py` (`pi0_libero`: `num_train_steps=30_000`).
