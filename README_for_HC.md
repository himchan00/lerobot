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

### 2.5 LIBERO + Latent-SDE + 손목 F/T (`libero_hf` 재생성 데이터)

데이터 재생성 (`examples/port_datasets/libero_hf/`, 이 PC의 Git Bash에서):

```bash
ssh gpu9 'bash -s' < examples/port_datasets/libero_hf/libero_hf_gpu9.sh                      # smoke (libero_spatial task0, demo 2개)
ssh gpu9 'STAGE=full WORKERS=8 bash -s' < examples/port_datasets/libero_hf/libero_hf_gpu9.sh  # 전체 (tmux libero_hf): download → replay → aggregate → slim
```

- raw: `yifengzhu-hf/LIBERO-datasets` 4개 suite hdf5 34 GB → `/PublicSSD/himchan/libero_hf/raw`. 2026-10-04에는 `snapshot_download`(xet)가 2 MB/s라 curl 6개 병렬로 받음 (약 10분).
- 재생 절차는 OpenVLA와 같음 (settle 10 step, no-op 제거, 성공 demo만). robosuite 1.4.0 + MuJoCo 3.8.1, 500 Hz 로깅. replay 약 25분 (worker 8개).
- 결과: 1,725 에피소드 / 279,278 프레임 / 40 태스크 (`lerobot/libero`는 1,693 / 273,465).

  | suite | spatial | object | goal | 10 |
  |---|---|---|---|---|
  | 성공 demo / 500 | 444 | 463 | 432 | 386 |
  | OpenVLA (`lerobot/libero`) | 432 | 454 | 428 | 379 |

- 출력 두 개:
  - `out/final` (`himchan00/libero_hf`): `hf.*`/`osc.*` 포함 전체. ODE 단계용.
  - `out/policy` (`himchan00/libero_hf_policy`): policy 학습용. `final`에서 `hf.*`/`osc.*`/`raw.*` 열만 빼고 영상은 symlink.
    lerobot이 여러 프레임 창을 읽을 때 행 전체를 가져와서, 넓은 `hf.*` 열이 있으면 샘플당 162 ms (슬림 13 ms, `lerobot/libero` 11 ms).
- `observation.ft_wrench` (25, 6): 직전 control step 동안 손목에 걸린 환경 상호작용 wrench.
  - F/T 센서 아래 바디에 걸린 MuJoCo 접촉력의 합이고, 중력·관성을 보상한 센서값에 해당. free space에서는 정확히 0.
  - EE 점 기준, physics step마다 EE frame으로 회전, 단위 N / N·m. 0번 프레임은 마지막 settle step 값.
  - 처음 버전(5r5euhpr)은 중력만 보상한 센서값이었음 (아래 결과). 그 데이터는 `/PublicSSD/himchan/libero_hf/out_ft_measured`에 보관.
  policy 입력이 될 수 있는 키는 `observation.*`뿐 (lerobot 전처리 `batch_to_transition`이 나머지를 버림). 그래서 `hf.*`/`osc.*`는 ODE 단계에서 쓸 때 경로가 따로 필요.
- 관측 시점: 로거는 rollout을 바꾸지 않지만 매 physics step 뒤 `forward()`를 불러, 관측이 stock env보다 physics step 하나 늦게 찍힘 (25 substep 중 24번째 vs 23번째, EE 위치 차이 0.5 mm 미만).
  그래서 이 데이터로 학습한 policy는 F/T 입력이 없어도 `--env.ft_wrench=true`로 평가.
- lerobot `aggregate_datasets`는 2차원(Array2D) feature를 pandas로 다시 쓰다가 깨짐 → `regenerate_libero_hf.py aggregate`에서 읽기/쓰기 함수를 바꿔 우회.

학습 (robot9 `outputs/queue_libero/queue16.sh`; 처음 버전과 control run은 `queue15.sh`):

```bash
lerobot-train \
    --policy.type=latent_sde \
    --policy.use_ft=true \
    --policy.z_dim=32 \
    --policy.state_noise_std=0 \
    '--policy.down_dims=[512,1024,2048]' \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --env.ft_wrench=true \
    --dataset.repo_id=himchan00/libero_hf_policy \
    --dataset.root=/PublicSSD/himchan/libero_hf/out/policy \
    --dataset.video_backend=torchcodec \
    --steps=30000 \
    --batch_size=64 \
    --save_freq=5000 \
    --env_eval_freq=10000 \
    --eval.n_episodes=1 \
    --eval.use_async_envs=false \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/latent_sde_so3_libero_hf_30k_z32_noise0_tokprior_zfilm_lin_dpwidth_ftext \
    --job_name=latent_sde_so3_libero_hf_30k_z32_noise0_tokprior_zfilm_lin_dpwidth_ftext \
    --wandb.enable=true \
    --wandb.project=lerobot_libero
```

- 비교 기준: `latent_sde_so3_libero_30k_z32_noise0_tokprior_zfilm_lin_dpwidth` (0tlfmmk9, `lerobot/libero`), 10k/20k/30k 평가 35 / 67.5 / 70%.
  데이터가 바뀌었으므로 같은 데이터에서 F/T만 뺀 control도 돌림 (`--policy.use_ft=true`를 빼고 이름에서 `_ft`를 뺌).
- F/T 입력: frame마다 `asinh([마지막 값, 평균] / c)` 12차원 (c = 1 N, 0.1 N·m). posterior는 tick마다 action 뒤에, drift는 state feature 뒤에 이어 붙임. prior는 그대로.
- 볼 지표:
  - `ft_usage_gap`: batch 안에서 F/T를 섞었을 때 recon 증가량. 0에 가까우면 drift가 F/T를 무시.
  - `z_usage_gap`: F/T가 z 대신 action을 설명하는 shortcut 때문에 z 사용이 줄어드는지.
- 5090, batch 64에서 0.342 s/step (기준과 같음), 메모리 약 11 GB.
- 결과 (30k, 학습 중 평가 40 에피소드):

  | run | 10k | 20k | 30k |
  |---|---|---|---|
  | F/T 센서값, 처음 버전 (`5r5euhpr`) | 0% | 2.5% | 5% |
  | F/T 접촉력, asinh (`2z2rhedi`, `_ftext`) | 27.5% | 65% | 67.5% |
  | F/T 접촉력, FORGE 전처리 (`e7fsi0qh`, `_ftforge`) | 35% | 45% | 75% |
  | control, 같은 데이터 (`re7rhes8`) | 47.5% | 52.5% | 70% |
  | 기준, `lerobot/libero` (`0tlfmmk9`) | 35% | 67.5% | 70% |

- 원인은 copycat. F/T가 직전 action의 판독값이 되고, drift가 그것을 베낌.
  - 자유공간(프레임의 80%)에서 중력 보상한 손목 F/T는 그리퍼(0.52 kg)의 관성 반력이고, 이 힘은 직전 명령이 정한 OSC 가속도에서 나옴.
  - 데이터에서 측정 F/T로 a(t-1)을 맞히는 held-out R²: 자유공간 위치 0.94 / 회전 0.73 (GT 접촉력으로는 0.13).
  - demo action이 매끄러워서 (a(t-1) → a(t) R² 0.98) F/T만으로 a(t)를 맞히는 R²가 0.89로, state(0.67)보다 높음.
- 학습 지표 (30k, F/T vs control):
  - recon 0.0027 vs 0.027
  - σ_grip 0.020 vs 0.18
  - `z_usage_gap` 0.38 vs 1.38
  - `ft_usage_gap` 1.2 (F/T를 섞으면 오차가 목표 분산보다 커짐)
- 폐루프 (spatial task 0, 1):
  - F/T가 이제 policy 자신의 직전 action을 반영해 자기강화 루프가 됨.
  - 그리퍼가 에피소드당 36 / 49번 열고 닫힘 (control은 1번). 그릇 위에서 280 step 진동하고, 일부 구간은 평균 34 N으로 누름.
  - F/T만 0으로 바꾸면 |a_pos|가 0.5~0.9에서 0.05~0.17로 떨어짐. 움직임 대부분이 F/T에서 나옴.
- 조치: `observation.ft_wrench`를 환경 상호작용 wrench(접촉력, free space에서 0)로 바꿔 재생성하고 `..._ftext`로 다시 학습 (queue16).
  - demo는 같으므로 control `re7rhes8`은 그대로 비교 기준.
  - 접촉력으로는 free space에서 a(t-1)을 맞히는 R²가 0.13.
- 남은 위험: 직전 명령 정보가 일부 남음 (접촉 구간 R² 0.51). `ft_usage_gap`과 `z_usage_gap`으로 확인.
  - 물체를 들고 옮길 때는 물체 무게와 관성(m_obj·a)이 남음.
  - 접촉 중에는 임피던스 때문에 접촉력이 직전 명령을 담음.
- 그래도 안 되면: F/T를 posterior/prior에만 넣기, drift 입력 F/T에 dropout이나 dead-band 적용, impedance ODE의 외력 항으로만 사용.
- take 2 (`..._ftext`, `2z2rhedi`) 10k: 27.5% (control 47.5%). 학습 데이터에는 control보다 잘 맞지만 폐루프는 낮음.
  - 직전 action을 같이 주면 F/T가 다음 action에 주는 held-out 정보가 없음 (R² 0.986 → 0.985).
  - 접촉 구간 그리퍼는 학습 R² 0.904 → 0.952, held-out 0.827 → 0.772로 과적합.
  - LIBERO demo는 SpaceMouse 원격조작(힘 피드백 없음)이라, 힘에 반응한 action이 데이터에 거의 없음.
- take 3 (`..._ftforge`, queue17): take 2와 같고 F/T 전처리만 FORGE 방식 (`--policy.ft_preprocess=forge`, dropout 없음).
  - EMA: Isaac Lab FORGE와 같은 시간상수 29 ms (500 Hz에서 α 0.0667), 최근 3 control step.
  - 학습 때 잡음 1 N / 0.1 N·m (FORGE는 힘에 1 N).
  - `tanh(w / (260 N, 12 N·m))`: 특징 표준편차 0.03~0.08로 drift state 특징(0.01~0.18)과 같은 스케일.
  - take 2의 asinh 특징은 표준편차 0.8~1.4로 state의 약 10배였음.

### 2.6 MimicGen F/T 진단 (2026-10-04, 학습 없음)

F/T가 중요한 sim benchmark 후보인 MimicGen에서, demo action에 F/T 정보가 있는지 LIBERO와 같은 방법으로 확인.

- 데이터: `amandlek/mimicgen_datasets`.
  - core D1 4개 (square, threading, coffee, three_piece_assembly, 각 1,000 demo)
  - source (사람 demo 10개씩)
  - 위치: `/PublicSSD/himchan/mimicgen_data/{core,source}`
- 환경: MimicGen 권장대로 robosuite `b9d8d3de`(1.4.1) + MuJoCo 2.3.2.
  - 경로: `/PublicSSD/himchan/envs/mimicgen` (gpu9 home이 이미 113 GB라 PublicSSD에 둠).
  - mimicgen 코드는 설치하지 않고 `PYTHONPATH=/PublicSSD/himchan/mimicgen`로 씀.
  - latent_sde env(robosuite 1.4.0, MuJoCo 3.8.1)에서는 coffee가 뜨지 않고(1.4.1 API 없음), 한 스텝 재생 오차도 약 1000배 큼.
- F/T 복원 (`mimicgen_data/tools/replay_ft.py`):
  - demo마다 `states[t]`로 되돌린 뒤 `actions[t]`를 한 스텝 실행하고, `libero_ft.FTWrenchLogger`로 접촉 wrench를 기록 (libero_hf와 같은 정의).
  - 한 스텝 재생 오차: EE 위치 중앙값 1e-10 m. 재생 시간은 task당 2~6분 (worker 30개).
  - OSC nullspace 목표: MimicGen 생성 데이터는 에피소드 시작 관절값 (`--nullspace start`, 기본), source 사람 demo는 reset 자세 (`--nullspace reset`).
- 분석 (`tools/ft_info.py`):
  - LIBERO held-out 검사와 같은 3×512 MLP를 쓰고, 검증 에피소드로 early stopping.
  - 입력: state(t), state(t-1), a(t-1).
  - 로그: `ft/info_libero.log`, `ft/info_mg.log`.
- 결과: F/T를 더했을 때 test R² 변화 (접촉 프레임, 위치 / 회전).

  | 데이터 | 프레임 | 접촉 >5 N | a(t) | a(t) − a(t-1) |
  |---|---|---|---|---|
  | LIBERO 4 suite | 277,553 | 15.8% | −0.002 / −0.003 | −0.005 / −0.003 |
  | Square D1 | 151,400 | 26.6% | −0.001 / −0.024 | −0.005 / −0.007 |
  | Threading D1 | 222,115 | 2.2% | −0.025 / −0.052 | −0.060 / −0.081 |
  | Coffee D1 | 223,403 | 12.8% | −0.010 / −0.048 | −0.046 / −0.036 |
  | Three Piece D1 | 333,869 | 16.8% | +0.002 / −0.008 | −0.005 / +0.001 |

  - 물체 자세(비전 대용)를 입력에 더해도 같음.
  - 전체 프레임에서도 위치/회전 변화는 ±0.015 이내.
  - 그리퍼 차원은 접촉 프레임이 적은 threading에서 ±0.2~0.7로 흔들리고 방향이 일정하지 않음.
- 원인: MimicGen action은 사람 source 구간의 목표 자세를 물체 좌표로 옮긴 뒤 P 추종으로 만듦: `a = clip((x_target − x_ee) / 0.05) + N(0, 0.05²)` (`mimicgen/datagen/waypoint.py`).
  - action에 힘에 반응하는 항이 없음.
  - task 성공에 접촉이 중요해도, 접촉은 어떤 demo가 성공으로 남는지만 거름.
- 결론: F/T에 반응하는 policy를 BC로 시험하려면, 시연자(스크립트 전문가 또는 RL 전문가)가 F/T에 반응해야 함. 기존 sim demo(LIBERO, MimicGen)에는 그런 action이 없음.

### 2.7 Square-FT: F/T에 반응하는 스크립트 전문가 demo (2026-10-04)

robosuite Square(MimicGen Square D1 배치)를 F/T 없이는 끼우기 어렵게 바꾼 task (`src/lerobot/envs/square_ft.py`, `--env.type=square_ft`).

- task:
  - 페그 한 변을 41 mm로 키움 (너트 구멍 45.5 mm, 한쪽 틈새 2.25 mm).
  - 에피소드마다 페그의 충돌 박스만 그려진 박스에서 L∞ 4~8 mm 옮김. 카메라로는 구멍 위치를 알 수 없음.
  - 너트–페그 마찰 0.3 (기본 1 / 0.95). 손가락 패드(마찰 2)로 잡는 데는 영향 없음.
  - OSC_POSE, kp 150, 20 Hz, 관측은 LIBERO와 같음 (카메라 2대 256, state 8, `observation.ft_wrench` (25, 6)). 언어: "put the square nut on the square peg".
  - 주의: MuJoCo 3는 고정 바디 geom의 월드 자세를 다시 계산하지 않음. 그래서 오프셋은 컴파일 후 `geom_pos`가 아니라 XML(`_load_model`)에 넣음. 앞의 방식은 효과가 없었음.
- 전문가 (`examples/port_datasets/square_ft/generate_square_ft.py`):
  - 참 너트 자세와 그려진 페그만 보고, 숨은 오프셋은 모름.
  - 너트를 그려진 페그 위로 옮겨 내리다가, 페그 윗면에 닿으면 2 step 눌러 지지력의 CoP(토크/힘)를 읽음.
  - 그 뒤 윗면에 닿은 채로 CoP 쪽으로 미끄러뜨리고, 매 step 현재 렌치로 CoP를 다시 읽음. 지지력이 사라지면(0.8 N 미만) 삽입.
  - CoP는 항상 구멍의 페그 쪽에 있어서, 매 step의 행동이 현재 F/T의 함수임.
  - 성공률 약 75% (실패는 잡기 실패, 삽입 중 걸림 등). 성공 에피소드만 저장.
- F/T 정보 검사 (전문가 demo 1,200개 중 성공분, 테스트 에피소드 177개, `mimicgen_data/tools/ft_phase.py`): 미끄러뜨리기 프레임의 다음 action 위치 R².

  | 입력 | F/T 없음 | F/T 있음 |
  |---|---|---|
  | state | −0.11 | 0.45 |
  | state + a(t-1) | 0.59 | 0.69 |
  | state(t, t-1) + a(t-1) | 0.60 | 0.68 |

  - 방향을 처음 고르는 step에서는 0 → 0.16~0.19.
  - LIBERO와 MimicGen에서는 같은 검사에서 이득이 0 이하였음 (2.6).
  - 시행착오: 누른 뒤 들어 올려 옮기는 전문가는 옮길 때 F/T가 0이라, 직전 action이 있으면 F/T 이득이 없었음 (velocity-blind drift는 따라 할 수 없음).
- 데이터: 1,000 에피소드, 201,475 프레임, 전문가 시도 1,337번 중 성공분 (`generate_square_ft.py shards --episodes 1000 --workers 16 --seed 500000`, 이어서 `aggregate`, 48분) → `/PublicSSD/himchan/square_ft/data/final` (`himchan00/square_ft`, fps 20).
  - CPU 렌더링(OSMesa, `LP_NUM_THREADS=1`)으로 생성. EGL과의 차이는 평균 0.8 / 0.45 단계 (영상 압축 오차보다 작음).
  - `expert.phase`, `expert.peg_offset` 열은 분석용 (policy 입력 아님).
- 학습 (`outputs/queue_square_ft/queue.sh`, tmux `latent_sde:squareft`): `_ftforge` (F/T, FORGE 전처리)와 control을 5090에서 동시에, 30k step, 10k마다 50 에피소드 평가. wandb `himchan00/lerobot_square_ft`.
  - latent_sde preset은 LIBERO와 같음 (`ENV_PRESETS["square_ft"]`).
  - run: `_ftforge` 6n6wzif6, control sparm4d3. 10-04 22:54 시작, 두 run 동시에 약 0.6 s/step, 10-05 05시경 종료 예상.
  - 데이터 생성은 worker당 RAM 약 2.5 GB라 gpu9(60 GB)에서는 16개까지 (28개로 돌렸을 때 메모리 부족으로 중단됨).
- 결과 (학습 중 평가, 50 에피소드, GPU 렌더링):

  | step | `_ftforge` 6n6wzif6 | control sparm4d3 |
  |---|---|---|
  | 10k | 8% | 16% |
  | 20k | 18% | 26% |
  | 30k | 38% | 22% |

  - 학습 지표는 거의 같음: recon 0.030 / 0.031, prior recon 0.040 / 0.040, `z_usage_gap` 1.17 / 1.18. `ft_usage_gap` 0.0015 (LIBERO FORGE run은 0.06).
- 계측 평가 (30k, 같은 seed 10000~10099, CPU 렌더링, `/PublicSSD/himchan/square_ft/diag/diag_eval.py` + `analyze_diag.py`):

  | 조건 | 성공 | 잡음 | 그려진 페그 위 | 윗면 착지 | 착지 후 삽입 | 윗면 체류 중앙값 |
  |---|---|---|---|---|---|---|
  | F/T policy | 32% | 75% | 68% | 64% | 50% | 44 step |
  | F/T policy, 평가 때 F/T = 0 | 19% | 72% | 59% | 58% | 33% | 99 step |
  | control | 26% | 79% | 67% | 66% | 39% | 68 step |

  - F/T를 0으로 넣으면 성공이 32% → 19% (짝지은 McNemar p = 0.019). 윗면에서 옆으로 움직이는 step 비율도 73% → 17%로 떨어짐. policy가 F/T를 접촉 감지 신호로 씀.
  - 하지만 방향은 못 읽음: 윗면에서의 수평 명령과 숨은 구멍 방향의 cos 중앙값 −0.12 (양수 비율 48%, control은 0.19 / 58%).
  - F/T vs control은 짝지은 비교에서 19 vs 13 (p = 0.38). 학습 중 평가와 합치면 34% vs 25%.
  - 공통 병목: 잡기 실패 21~28%.
- 해석: FORGE 스케일이 LIBERO용(260 N, 12 N·m, 노이즈 1 N / 0.1 N·m)이라, 결정 구간(힘 중앙값 6.8 N, 토크 0.33 N·m)의 F/T 특징 표준편차가 0.005~0.04로 state 위치 특징(0.05~0.13)보다 5~20배 작고, 신호 대 노이즈 비도 0.6~5. 접촉 유무는 보이지만 CoP(토크/힘 비)는 묻힘.
- 조치 (10-05): F/T 스케일을 학습 데이터에서 계산 (`policies/latent_sde/ft_scale.py`, `action_scale`과 같은 방식).
  - `--policy.ft_scale`을 주지 않으면, 처음부터 학습할 때 FORGE EMA를 거친 F/T의 접촉 프레임(|F| > 0.5 N)에서 축별 |w|의 p90을 스윕해 `config.ft_scale`에 저장하고 캐시함. 학습 노이즈는 `ft_noise`(기본 0.05) × 스케일.
  - 값: Square-FT 힘 3.7 / 4.0 / 12.2 N, 토크 0.63 / 0.24 / 0.16 N·m. LIBERO 22 / 36 / 73 N, 1.0 / 3.3 / 1.8 N·m.
  - 이전 run(6n6wzif6, LIBERO의 e7fsi0qh)을 다시 평가할 때는 `--policy.ft_scale=[260,260,260,12,12,12]`.
  - run: `..._ftdata` = 5kmn2jce (`outputs/queue_ft/queue3.sh`).

### 2.8 Wipe-FT: 숨은 표면에서 힘을 맞춰 닦기 (2026-10-05)

Square-FT가 F/T로 "어디로"(구멍 방향)를 읽는 task라면, Wipe-FT는 "얼마나 세게"(수직 힘)를 매 step 맞추는 task (`src/lerobot/envs/wipe_ft.py`, `--env.type=wipe_ft`).

- task: robosuite Wipe (robosuite 벤치마크 9개 task 중 하나), Panda + WipingGripper (손가락 없는 12×5 cm 패드), OSC_POSE kp 150, 20 Hz.
  - 테이블 충돌면만 그려진 테이블에서 높이 ±15 mm, roll/pitch ±2° 숨겨서 옮김. 카메라에는 평평한 원래 테이블만 보임.
  - 마커 40개 (robosuite 경로 생성, 5 mm 간격, 약 20 cm). 도구가 마커를 덮고 수직력이 5~20 N일 때만 닦임 (robosuite 원래 판정은 위치만 봄). 35 N 넘게 누르면 실패로 끝남. 전부 닦으면 성공.
  - 측정해 보니 OSC 목표를 표면 아래로 9 mm 두면 19 N, 21 mm면 36 N. 닦이는 범위가 목표 깊이로 약 7 mm 폭이라 숨은 높이(±15 mm + 기울기)를 모르면 맞추기 어려움.
  - 관측: Square-FT와 같음. 그리퍼 state는 0, action은 7차원 중 마지막을 무시 (데이터에는 −1).
- 전문가 (`examples/port_datasets/wipe_ft/generate_wipe_ft.py`): 마커 경로는 알고 숨은 표면은 모름.
  - 첫 마커 위에서 내려가다 (그려진 표면 3.5 cm 위부터 3 mm/step) F/T 위쪽 힘이 2 N을 넘으면 접촉.
  - 경로를 8 mm/step으로 따라가며 매 step 높이 목표를 `z_tgt -= 0.3 mm/N × (10 N − F)` (step당 ±2 mm 제한)로 갱신. 수직 명령이 매 step 현재 F/T의 함수.
  - 경로 끝에서 덜 닦인 마커가 있으면 한 번 되돌아 닦음.
  - 성공률 약 85% (실패는 착지 지연, 끝 마커 미닦기 등). 닦는 동안 힘 p10/p50/p90 5.2 / 9.4 / 12.7 N.
- F/T 정보 검사 (전문가 demo 600개 중 성공분, 테스트 에피소드 102개, 닦기 프레임 6,250개, `mimicgen_data/tools/wphase.py`): 다음 action 수직 성분 R².

  | 입력 | F/T 없음 | F/T 있음 |
  |---|---|---|
  | state | 0.04 | 0.53 |
  | state + a(t-1) | 0.91 | 0.91 |
  | state(t, t-1) + a(t-1) | 0.93 | 0.91 |

  - 현재 프레임만 보는 drift에는 큰 정보. 하지만 직전 action을 알면 정보가 없음: 준정적 접촉에서 F ≈ K(x_d − x)라 자기 명령과 위치로 힘을 계산할 수 있음. Square-FT는 접촉점(토크) 덕분에 직전 action을 알아도 +0.08이 남았음.
- 데이터: 1,000 에피소드 (`generate_wipe_ft.py shards --episodes 1000 --workers 16 --seed 600000`, 이어서 `aggregate`) → `/PublicSSD/himchan/wipe_ft/data/final` (`himchan00/wipe_ft`, fps 20).
- 학습 (`outputs/queue_ft/queue3.sh`, tmux `latent_sde:squareft`, wandb `himchan00/lerobot_wipe_ft`): Square-FT `_ftdata` (5kmn2jce)와 Wipe-FT `_ftdata`를 동시에, 이어서 Wipe-FT control. Wipe run은 CPU smoke test가 통과해야(`/PublicSSD/himchan/wipe_ft/SMOKE_OK`) 시작.
  - 10-05 18:11에 큐를 바꾸려고 tmux 창에 Ctrl-C를 보내 먼저 시작된 Square `_ftdata`(gd4d9vgm, 13분)를 끊었음. 같은 설정으로 18:12에 다시 시작 (5kmn2jce).

### 2.9 Door-FT: 경첩 쪽이 숨은 문 열기 (2026-10-05)

robosuite Door (`src/lerobot/envs/door_ft.py`, `--env.type=door_ft`). F/T가 묶인 운동의 방향(어느 쪽으로 열리는지)을 알려 주는 task.

- task: 손잡이를 문 중앙의 대칭 막대로 옮기고, 실제 경첩을 에피소드마다 왼쪽 또는 오른쪽 기둥에 숨겨 둠 (두 기둥은 똑같이 보임). 걸쇠 없음.
  - 열림 범위 0~1.6 rad, 0.7 rad 넘게 열면 성공. 손–문 순 힘이 100 N을 넘으면 실패로 끝남 (똑바로만 당겨 0.7 rad까지 열면 옆 어긋남 약 7.5 cm, 약 110 N).
  - 원래 Door는 경첩이 오른쪽 기둥에 고정되어 영상과 문 위치로 알 수 있음.
- 전문가 (`examples/port_datasets/door_ft/generate_door_ft.py`): 손잡이 자세와 문 형상(경첩이 중심에서 0.255 m)은 알고, 경첩 쪽은 모름.
  - 막대 받침 옆을 앞에서 잡고(손가락 수직), 8 step 똑바로 당기며 옆 방향 F/T로 경첩 쪽을 판정 (왼쪽 약 −7 N, 오른쪽 +1~4 N, 기준 −2.5 N).
  - 판정한 경첩을 중심으로 0.04 rad/step씩 호를 따라 당김. 옆 힘이 25 N을 3 step 넘으면 반대쪽으로 바꿈. 당기는 동안 그리퍼 yaw 고정 (yaw를 따라 돌리면 막대가 비틀려 힘이 튐).
  - 성공률: 오른쪽 79%, 왼쪽 22% (왼쪽은 0.8 rad 근처에서 팔이 한계에 닿음). 데이터는 좌우 500개씩 맞춤.
- 데이터: 1,000 에피소드 (`generate_door_ft.py shards --episodes 1000 --workers 20 --seed 700000`, gpu17 CPU) → `/PublicSSD/himchan/door_ft/data/final` (`himchan00/door_ft`).

### 2.10 세 F/T 벤치마크 × 세 policy (2026-10-05 밤, gpu9/17/19, 10-06 오전까지)

| | latent_sde + F/T (데이터 기반 스케일) | latent_sde, F/T 없음 | SmolVLA |
|---|---|---|---|
| Square-FT | 5kmn2jce (gpu9) | sparm4d3 (완료) | tld3q4g2 (gpu19) |
| Wipe-FT | jex7tu6k (gpu9) | gpu9 | dh59wfjm (gpu19) |
| Door-FT | dqzh7cu4 (gpu17) | p2dutj2a (gpu17) | gpu19 |
| FT-3 (2.11) | gpu17 | gpu9 | gpu19 |

- 모두 30k step, batch 64, 10k마다 50 에피소드 평가. SmolVLA는 2.3절 레시피 (SmolVLM2 가중치, 실행 10 step).
- 큐: gpu9 `outputs/queue_ft/queue3.sh`, gpu17 `queue_door.sh`, gpu19 `queue_sv.sh` (각 서버 tmux). 상태는 `outputs/queue_ft/status*.txt`.
- gpu19 환경: `~/miniconda3/envs/latent_sde` (gpu9의 `pip freeze`로 버전 고정). `/tmp/robosuite.log`가 다른 사용자 파일이라 robosuite `macros_private.py`에서 파일 로그를 끔.
- 서버끼리 ssh가 안 돼서 데이터셋은 이 PC를 거쳐 `ssh A tar | ssh B tar`로 옮김.
- gpu17에는 OSMesa가 없어서 생성·smoke·평가 모두 `MUJOCO_GL=egl`.

### 2.11 FT-3: 세 F/T 과제를 한 policy로 (LIBERO식 multi-task, 2026-10-05 밤)

- 데이터 `himchan00/ft3` (`/PublicSSD/himchan/ft3/data/final`, gpu9/17/19 모두): Square-FT + Wipe-FT + Door-FT 각 1,000 에피소드 = 3,000 에피소드, 568,647 프레임.
  - task 0/1/2 = "put the square nut on the square peg" / "wipe the marked path on the table" / "open the door".
  - 과제별 숨은 라벨(`expert.peg_offset`, `expert.surface`, `expert.hinge_side`)은 빼고 `expert.phase`만 남김. 원본과 프레임·F/T·action이 같은지 확인함.
  - 데이터 기반 F/T 스케일: [2.6, 6.5, 12.1, 0.87, 1.06, 0.55].
- 평가 `--env.type=ft3`: LIBERO suite처럼 과제마다 50 에피소드, 에피소드 길이는 과제별 원래 값(400/300/350).
  - wandb에는 전체 성공률만 올라감. 과제별 성공률은 학습 로그의 `Suite per_task aggregated` 줄에 있음.
- 학습 설정은 2.10과 같음. wandb `himchan00/lerobot_ft3`.
- 큐 `outputs/queue_ft/queue_ft3.sh {ctrl|ft|smolvla}` (각 서버 tmux 창 `ft3`, 상태 `status_ft3.txt`).
  - gpu9은 F/T 없음, gpu17은 F/T, gpu19는 SmolVLA.
  - 각 서버의 단일 과제 run 하나가 끝나면 시작.

```bash
# 데이터셋 (세 데이터셋이 있는 서버에서; gpu19에서 만듦)
python examples/port_datasets/ft3/build_ft3.py --out /PublicSSD/himchan/ft3/data \
    --sources /PublicSSD/himchan/square_ft/data/final /PublicSSD/himchan/wipe_ft/data/final /PublicSSD/himchan/door_ft/data/final
# latent_sde + F/T (F/T 없음: use_ft/ft_preprocess 빼기. SmolVLA: 2.3 명령에 --env.type=ft3, 같은 dataset)
lerobot-train --policy.type=latent_sde --policy.use_ft=true --policy.ft_preprocess=forge --policy.z_dim=32 \
    --policy.state_noise_std=0 '--policy.down_dims=[512,1024,2048]' --policy.push_to_hub=false --env.type=ft3 \
    --dataset.repo_id=himchan00/ft3 --dataset.root=/PublicSSD/himchan/ft3/data/final --dataset.video_backend=torchcodec \
    --steps=30000 --batch_size=64 --save_freq=5000 --env_eval_freq=10000 \
    --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false
```

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
