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

### 2.7 (지움) Square-FT, Wipe-FT, Door-FT, FT-3, FT-N, FT-4 v1 (2026-10-04~11)

숨은 오프셋(충돌 형상만 옮김)이나 원본과 다른 장면을 쓴 이전 F/T 벤치마크들이다. 영상과 물리가 어긋나거나 장면이 원본이 아니어서 2026-10-12에 코드와 데이터를 지웠다. 코드와 이 README의 기록(옛 §2.7–2.11)은 커밋 `6a305f32`에, 학습 기록은 wandb에 남아 있다.

### 2.8 FT-4: 지각 오차 아래의 네 robosuite 과제 (2026-10-12, gpu9)

목적은 force-reactive 데모가 필요한지 보는 것이다. latent_sde를 native / reactive 데모 × F/T 있음 / 없음으로 학습한다(2 × 2).

**장면** (`src/lerobot/envs/ft4.py`, `--env.type=ft4`): robosuite 원본 Square, Stack, Wipe(WipingGripper), Door(걸쇠 있음).
- F/T 때문에 바꾼 것은 Square 페그 폭 하나다(32 → 41 mm, 너트 구멍 45.5 mm라 틈새가 한쪽 2.25 mm).
- 평가 때 매 에피소드 카메라 보정을 틀어 지각 오차를 넣는다(`perturb_cameras`).
  - 3인칭 카메라: 축마다 ±5 mm, ±0.5°. 손목 카메라: ±2 mm, ±0.5°.
  - `--env.camera_error`로 크기를 조절하고, 0이면 끈다.
- 접촉 규칙(`contact_rules`):
  - Stack: 블록을 20 N 넘게 누르거나 3 mm 넘게 떨어뜨리면 실패.
  - Wipe: 5–25 N으로 고르게(압력 중심이 패드 가운데 60%) 눌러야 닦이고, 40 N을 넘으면 실패. robosuite의 조기 종료(팔이 테이블에 닿음, 관절 한계)도 실패.
  - Square와 Door는 robosuite 규칙 그대로다.

**데모** (`examples/port_datasets/ft4/`, 데이터 `/PublicSSD/himchan/ft4/data/<task>_native`, `<task>_conv`)
- native(`generate_ft4.py`): 지각이 정확한 비반응 전문가의 데모다. 전문가는 `square_expert.py`(Square)와 `experts.py`(Stack, Wipe, Door)에 있다.
  - 과제당 1000개. seed는 square 900000, door 910000, wipe 920000, stack 940000(split 16).
  - 실패가 특정 시작 조건에 몰리지 않게 고쳤다(10-12).
    - Square는 너트를 옮기며 손잡이가 로봇 쪽을 향하게 돌린다.
    - Door는 손가락 방향을 고정한다.
    - Wipe는 천천히 닿은 뒤 8 mm로 누른다.
  - 전문가 성공률: Square 98%, Stack 93%, Wipe 93%, Door 99%. 성공한 에피소드만 남긴다.
- conv(`convert_native.py`): native 데모를 지각 오차 데모로 바꾼 것이다.
  - native 에피소드를 seed chain에서 그대로 재생해 장면 전체를 얻는다(문 위치, Wipe 경로, gripper 명령, OSC 자세까지 정확히 일치).
  - 목표에 대해 움직이는 구간에서 힘 반응 전문가에게 넘긴다. 이 전문가는 목표를 δ만큼 틀리게 알아서, 빗나가게 닿은 뒤 F/T로 고친다. 실패하면 δ를 새로 뽑아 3번까지 시도한다.
  - δ 범위: Square 페그 xy L∞ 4–12 mm, Stack 옆 ±5 mm·높이 4–10 mm, Wipe 높이 ±15 mm·기울기 ±3°, Door 경첩 ±3 cm.
  - label: `expert.perception_error`, `expert.source`(native seed, k).
- 묶기(`build_ft4.py` + `recorder.views`): `/PublicSSD/himchan/ft4/data/ft4v2_native`, `ft4v2_reactive`(policy view, depth 없음).
- pilot(변환 성공 / 같은 오차에 반응 없음 / native 재생, 접촉 규칙 적용):

  | 과제 | 변환 성공 | 같은 오차, 반응 없음 | native 재생 |
  |---|---|---|---|
  | Square | 23/32 | 14/32 | 32/32 |
  | Stack | 40/40 | 1/40 | 40/40 |
  | Wipe | 24/24 | 0/24 | 24/24 |
  | Door | 32/32 | 32/32 | 32/32 (경첩 오차가 결과를 바꾸지 않는 대조군) |

```bash
cd examples/port_datasets/ft4   # latent_sde env, MUJOCO_GL=egl, D=/PublicSSD/himchan/ft4/data
python generate_ft4.py shards --task square_ft_native --episodes 1000 --splits 16 --workers 14 --seed 900000 --out $D/square_ft_native
python generate_ft4.py finish --out $D/square_ft_native --repo-id himchan00/square_ft_native
python convert_native.py pilot --task square_ft --per-worker 1 --workers 4    # replay check, conversion and its controls
python convert_native.py shards --task square_ft --workers 14 --out $D/square_ft_conv
python convert_native.py finish --out $D/square_ft_conv --repo-id himchan00/square_ft_conv
python build_ft4.py --sources $D/{square,stack,wipe,door}_ft_native/final --out $D/ft4v2_native --repo-id himchan00/ft4v2_native
python -c "import recorder; recorder.views('$D/ft4v2_native')"
```
recording worker 하나가 약 2.5 GB를 쓴다. gpu9(60 GB)에서는 녹화 작업을 한 번에 하나만 돌린다.

**학습** (gpu9, wandb `himchan00/lerobot_ft4`, 2026-10-12~13)
- latent_sde, 30k step, batch 64. 두 run을 같이 돌리면 각각 1.42 step/s이고, 평가까지 약 7시간 걸린다.
- 평가는 `ft4`(카메라 오차 1, 접촉 규칙)로 10k마다 과제당 50 에피소드다.
- run 이름: `latent_sde_ft4v2_{native,reactive}_{ft,noft}_30k`. F/T 없는 run은 `--policy.use_ft`를 뺀다.
- 아래 결과의 F/T run은 F/T를 FORGE 방식(3 tick EMA + noise + tanh, `--policy.ft_preprocess=forge`)으로 읽었다. 이 옵션은 2026-10-27에 지웠다. 지금 코드는 F/T를 tick 평균 W̄_k로 읽으므로(§2.9) 같은 명령이라도 F/T run은 결과가 다를 수 있다. wrench noise 기본값(§2.10)도 이 실험에는 없었으므로 `--policy.wrench_noise_std=0`으로 끈다.
- `--wandb.notes` 같은 문자열 인자에 `: `를 넣으면 draccus가 YAML로 읽어 시작 직후 죽는다.
```bash
lerobot-train --policy.type=latent_sde --policy.z_dim=32 --policy.state_noise_std=0 '--policy.down_dims=[512,1024,2048]' \
    --policy.use_ft=true --policy.wrench_noise_std=0 --policy.push_to_hub=false --env.type=ft4 \
    --dataset.repo_id=himchan00/ft4v2_reactive --dataset.root=$D/ft4v2_reactive/policy --dataset.video_backend=torchcodec \
    --steps=30000 --batch_size=64 --save_freq=5000 --env_eval_freq=10000 \
    --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false \
    --wandb.enable=true --wandb.entity=himchan00 --wandb.project=lerobot_ft4 \
    --output_dir=outputs/train/latent_sde_ft4v2_reactive_ft_30k --job_name=latent_sde_ft4v2_reactive_ft_30k
```

결과 (30k, 성공률 %, 과제당 50 에피소드, seed 1개):

| 데모 | F/T | Square | Stack | Wipe | Door | 평균 | wandb |
|---|---|---|---|---|---|---|---|
| native | 있음 | 52 | 44 | 50 | 82 | 57.0 | `2gws75gq` |
| native | 없음 | 22 | 48 | 8 | 82 | 40.0 | `v5zmqg6l` |
| reactive | 있음 | 48 | 66 | 38 | 68 | 55.0 | `4qx5x4s0` |
| reactive | 없음 | 52 | 38 | 2 | 86 | 44.5 | `eamw8hho` |

**추가 평가** (2026-10-13, 같은 30k 체크포인트): 카메라 오차 0과 1로 다시 평가했다. seed가 1000부터라 학습 중 평가와 장면이 같다.
```bash
lerobot-eval --policy.path=outputs/train/latent_sde_ft4v2_native_ft_30k/checkpoints/030000/pretrained_model --env.type=ft4 \
    --env.camera_error=0 --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false --output_dir=<out>
```
실패 단계는 매 step 과제 상태를 기록하는 임시 스크립트로 봤다(`/PublicSSD/himchan/ft4/analysis`, repo 밖).

성공률 %, 각 칸은 카메라 오차 0 / 1(재평가) / 1(학습 중 평가, 위 표):

| 데모 | F/T | Square | Stack | Wipe | Door | 평균 |
|---|---|---|---|---|---|---|
| native | 있음 | 56 / 52 / 52 | 50 / 44 / 44 | 50 / 58 / 50 | 84 / 80 / 82 | 60.0 / 58.5 / 57.0 |
| native | 없음 | 44 / 42 / 22 | 48 / 44 / 48 | 12 / 8 / 8 | 78 / 80 / 82 | 45.5 / 43.5 / 40.0 |
| reactive | 있음 | 54 / 48 / 48 | 62 / 70 / 66 | 24 / 32 / 38 | 72 / 64 / 68 | 53.0 / 53.5 / 55.0 |
| reactive | 없음 | 54 / 46 / 52 | 48 / 48 / 38 | 4 / 6 / 2 | 84 / 86 / 86 | 47.5 / 46.5 / 44.5 |

- 카메라 오차는 거의 영향이 없다. 평균이 −0.5~+2점만 바뀌고, Square만 네 run 모두 오차 0에서 조금 높다(평균 +5).
- F/T 이득도 오차 0에서 그대로다(native +14.5, reactive +5.5). F/T가 돕는 것은 지각 오차 보정이 아니라 접촉 제어다.
- 같은 체크포인트를 같은 장면에서 다시 평가해도 정책 샘플링 노이즈로 과제당 최대 20점 차이가 난다(native F/T 없음 Square 22 vs 42). 그래서 아래는 세 평가를 합쳐 본다(과제당 150 에피소드, 장면은 같은 50개).

| 세 평가 평균 | Square | Stack | Wipe | Door | 평균 |
|---|---|---|---|---|---|
| native + F/T | 53 | 46 | 53 | 82 | 58.5 |
| native, F/T 없음 | 36 | 47 | 9 | 80 | 43.0 |
| reactive + F/T | 50 | 66 | 31 | 68 | 53.8 |
| reactive, F/T 없음 | 51 | 45 | 4 | 85 | 46.2 |

reactive 데모가 돕지 않는 이유. conv 데모에는 접촉 전에 보이지 않는 틀린 목표(δ)를 겨냥하는 부분과 접촉 후 F/T로 고치는 부분이 섞여 있다. 정책은 앞부분을 무작위 조준 오차로 따라 하고, 뒷부분은 F/T로 배운다(아래는 오차 0과 1 재평가를 합친 100 에피소드).
- Square는 상쇄된다.
  - reactive 정책은 너트가 페그에 처음 닿을 때 중앙에서 7–8 mm(L∞ 중앙값) 벗어나 있다. native는 5 mm다. 오차 0에서도 같아서 δ를 따라 한 것이다.
  - 틈새(2.25 mm) 안으로 바로 닿는 비율은 1–7%로, native의 13–28%보다 낮다.
  - 대신 6–12 mm 벗어났을 때 성공률은 67–81%로, native의 25–64%보다 높다.
- Stack은 reactive + F/T만 +20점이다.
  - native 데모는 블록 1 mm 위에서 놓는다(놓을 때 0.6 N). 판단 시점에 접촉이 없으니 F/T로 배울 신호가 없다.
  - native + F/T는 100번 중 23번 떨어뜨리고 11번 짓누른다.
  - reactive + F/T는 닿은 뒤 놓아서 짓누름 0번, 떨어뜨림 4번이다. 잡은 뒤 성공률은 90%다(다른 조합은 55–68%).
- Wipe는 reactive + F/T가 −21점이다.
  - 접촉 품질(5–25 N 98–100%, 기울기 0.1° 안팎)과 마커 90개까지의 속도(약 365 step)는 native + F/T와 같다.
  - 실패한 에피소드는 마지막 마커 약 15개를 남기고 600 step 시간 초과로 끝난다(성공은 100개 전부). 원인은 못 찾았다.
  - F/T가 없으면 패드가 기울고 압력이 고르지 않다. Wipe에서 F/T의 이득은 대부분 여기서 온다.
    - native: 기울기 0.55°, 고른 압력 19–23%. 100번 중 관절 한계 23번, 40 N 초과 18번.
    - reactive: 기울기 0.4°, 고른 압력 31–40%. 대부분 시간 초과다.
- Door는 reactive + F/T가 −14점이다.
  - 실패는 네 run 모두 손잡이는 돌렸는데 문이 0.11 rad에서 멈추는 모양이다.
  - 이 실패가 reactive + F/T에서만 두 배다. 원인은 못 찾았다.

결론: native 재생 + F/T가 가장 좋다(58.5). native 전문가의 판단 시점에 접촉이 있어야 F/T가 배울 신호가 생긴다(Stack). 확인된 것은 이 변환(접촉 전에 틀린 목표를 겨냥)이 돕지 않는다는 것까지다. 접촉 전 행동을 참 목표 쪽으로 라벨링하는 DART 방식은 시험하지 않았다. 학습 seed는 하나다.

### 2.9 Latent-SDE body mode의 frame과 Tier-3 impedance 출력 (2026-10-14~)

기호
- W: world.
- G: grip site(TCP, 손가락 끝 사이) frame. controller와 F/T가 쓰는 frame이다.
- G_0: chunk 시작(context와 z를 새로 뽑는 시점, 10 tick마다) 때의 G. G_t: 지금 tick의 G.
- H: 손 body(`robot0_right_hand`) frame. robosuite의 `robot0_eef_quat`가 이 자세다.
- G = H·Rz(−90°)이다(FT-4와 LIBERO의 Panda 그리퍼 공통). 위치는 둘 다 grip site 점이다.

| 대상 | 들어가는 것 | frame |
|---|---|---|
| 데이터 `observation.state` | 위치, 자세(axis-angle), finger | 위치 W, 자세 H. policy 입구(`forward`, `select_action`)에서 `hand_to_grip_site`로 자세를 G로 한 번 바꾼다 |
| 데이터 `observation.ft_wrench` | 환경이 그리퍼에 주는 wrench(grip site 점 기준) | 2 ms sample마다 그 순간의 G. policy는 tick 평균 W̄_k를 G_k로 돌려 읽는다(`geometry.tick_mean_wrench`) |
| Prior 입력 | 이미지 + chunk 시작 state | W에서 본 G_0의 위치와 자세, finger 절대값 |
| Posterior 입력(학습만) | tick마다 깨끗한 1-step 라벨 | OSC 출력이면 body increment / s + gripper, 그리고 W̄_k(`use_ft`). impedance 출력이면 [x̂_{k+1}, W̄_{k+1}] / s + gripper(F/T 없이, 라벨에 이미 들어 있다) |
| Drift 입력 | 지금 state + F/T + z | state는 G_0 기준 상대값, F/T는 W̄_k(G_t, `use_ft`; OSC와 impedance가 같은 입력과 scale `ft_scale`) |
| Drift 출력(OSC) | 다음 step body increment + gripper | G_t. OSC에 넣을 때 W 명령으로 바꾼다 |
| Drift 출력(impedance, `--policy.impedance_targets=true`) | 1-step likelihood: x̂_{t+1}(Gaussian), W̄_{t+1}(hurdle: 접촉 Bernoulli × Gaussian), 접촉 logit, gripper logit (14) | G_t. 배포 때 모델이 한 step 더 가서 x̂_{t+2}, W̄_{t+2}를 만들고 25개로 controller에 넣는다 |
| Tier-3 impedance controller 입력(`envs/ft4_impedance.py`, `--env.impedance=true`) | x̂_{t+1}, x̂_{t+2}, W̄_{t+1}, W̄_{t+2} | G_t 기준 상대값. 받자마자 W reference로 복원하고, 계산은 모두 W다 |

**Impedance 출력의 라벨** (`geometry.impedance_one_step`)
- x̂_{t+1}: 다음 frame에서 실제로 잰 pose를 G_t 기준 body increment로 나타낸 것이다.
- W̄_{t+1}: frame t+1의 `ft_wrench` 25개 sample 평균(tick 평균)을 G_t로 돌린 것이다. sample마다 기록된 physics step의 frame이 다르므로, 앞뒤 frame 사이를 slerp한 frame으로 각각 돌린 뒤 평균낸다.
- 접촉 라벨은 W̄_{t+1} ≠ 0이고, gripper 라벨은 command > 0이다. W̄의 Gaussian 항은 접촉 tick에만 loss에 들어간다.
- 에피소드 끝: 데이터에는 마지막 action 뒤의 state가 없다.
  - 그 한 frame은 마지막 두 frame의 속도를 유지해 외삽하고(데이터로 잰 오차 0.15 mm), wrench는 마지막 frame 값을 그대로 쓴다.
  - window가 에피소드의 마지막 frame에서 시작하면 외삽할 앞 frame이 없다. 그래서 그 tick의 x̂는 loss와 posterior에서 뺀다(sample의 약 0.3%). W̄와 gripper는 그대로 학습한다.
- 정규화: 위치·회전·힘·토크 4묶음마다 데이터 std로 나누고, σ²도 묶음마다 하나다. reactive 데이터에서 2.27 mm, 0.51°, 10.1 N, 0.53 N·m다. 두 logit은 BCE로 학습한다.

**배포** (모델 → controller)
- 모델이 x̂_{t+1}, W̄_{t+1}을 낸 뒤, 그 상태에서 한 step 더 가서 x̂_{t+2}, W̄_{t+2}를 만든다. 이 두 번째 step의 입력 W̄_{t+1}은 1[π > 0.5]·μ_W다(접촉 확률이 낮으면 0).
- controller에는 25개 값을 넘긴다: x̂_{t+1}, x̂_{t+2}, W̄_{t+1}, W̄_{t+2}(μ_W), gripper. 측정 힘 |W̄_t|가 `wrench_gate`(1 N) 미만이면 두 W̄를 0으로 둔다.
- controller(`envs/ft4_impedance.py`)는 매 tick 측정 상태에서 다음 목표까지 quintic reference를 만든다. F_d는 tick 중간(h/2)에서 W̄_{t+1}을 지나는 꺾은선이다(측정 W̄_t → W̄_{t+1} → (W̄_{t+1}+W̄_{t+2})/2).
- reset 직후 첫 tick은 직전 tick이 없다. 그래서 x̂_{t+1}을 측정 pose에 대해 같은 속도로 거꾸로 이어 직전 tick을 만들고, 시작 속도와 가속도를 구한다. FT-4는 첫 tick이 settle(정지 유지)이라 이 값들과 F_d가 모두 0이다.
- controller 세팅(aligned, κ = 1, full Λ)과 그 근거는 §2.10에 있다.

**F/T 입력** (OSC와 impedance 공통): drift는 W̄_k를 `ft_scale`로 나눠 읽는다. `ft_scale`은 impedance one-step W̄_{t+1} 라벨의 데이터 std이고, OSC run도 같은 데이터에서 같은 값을 쓴다. OSC posterior는 action과 함께 W̄_k를 읽는다.

**이 코드로 불러오지 못하는 체크포인트**
- 2026-10-14 이전(자세가 H frame): commit `02005dd5`의 코드로 평가한다.
- 2026-10-16 이전의 2-step impedance(25출력).
- 2026-10-27에 뺀 옵션으로 학습한 것:
  - policy: σ² 비례 모델(`sigma_model=proportional`), `prior_history`·`drift_history`, state 궤적 posterior(`posterior_commands=false`), `wrench_estimate`.
  - controller: κ < 1, 직전 reference 이어 붙이기, decoupled Λ.
- F/T를 asinh나 FORGE 방식(`ft_preprocess`)으로 읽은 OSC·LIBERO run. 지금은 F/T를 tick 평균 W̄_k로만 읽는다.
- `ft_scale`이 config에 없는 impedance 체크포인트는 `ft_scale = action_scale[6:12]`을 config에 써 넣으면 그대로 동작한다.

### 2.10 Tier-3 impedance: 고정한 세팅과 wrench noise 결과 (2026-10)

기호(x̂, W̄, π, μ_W, G)는 §2.9, native / reactive 데이터는 §2.8을 따른다. 아래 세팅이 2026-10-27 코드의 기본값이다.

**평가 방법.** 이 절의 /240 숫자는 모두 다음 명령을 S = 1000, 2000, 3000으로 한 번씩 돌린 합이다. 과제마다 20 × 3 = 60 에피소드이고, 네 과제를 합하면 240이다.

```bash
lerobot-eval --policy.path=<checkpoint>/pretrained_model --env.type=ft4 --env.impedance=true \
  --env.camera_error=1.0 --env.contact_rules=true \
  --eval.n_episodes=20 --eval.batch_size=10 --eval.use_async_envs=false --seed=S --policy.device=cuda
```

- `camera_error=1.0`: 에피소드마다 카메라 위치와 각도를 흔든다(agentview ±5 mm·±0.5°, wrist ±2 mm·±0.5°).
- `contact_rules=true`: Stack에서 블록을 짓누르거나 떨어뜨리면 실패다. Wipe에서 40 N을 넘거나 팔이 테이블에 닿아도 실패다. Wipe 마커는 패드 힘이 5–25 N이고 압력 중심이 패드 가운데 60% 안일 때만 지워진다.
- OSC policy는 `--env.impedance=true`를 빼고 같은 명령으로 잰다.
- 유의성은 두 비율의 z 검정이다. |z| ≥ 2를 유의로 본다.
- "학습 중 eval"은 학습하면서 10k step마다 같은 env로 과제당 50 에피소드(seed 1000부터)를 돈 결과다. 에피소드가 달라서 /240 숫자와 직접 비교하지 않는다.
- 학습 seed는 run마다 하나다.

**고정한 세팅**

| 부분 | 세팅 |
|---|---|
| 데이터 | `himchan00/ft4v2_reactive`. reactive 데모는 native보다 학습 중 eval이 약 2배다(30k: native 12.0% → reactive 22.5–27.5%, z_dim 8–64. 대부분 Stack에서 오르고, 아래 aligned controller 이전 값이다). |
| 출력 | one-step likelihood 14개(§2.9). σ²는 위치·회전·힘·토크 묶음마다 상수다. |
| 학습 noise | wrench noise σ 0.3, β 0.3이고 state noise는 없다(아래 "Wrench noise"). |
| z | prior에서 샘플하고 10 tick마다 새로 뽑는다. |
| 배포 W̄ | 모델의 두 번째 step 입력 W̄_{t+1}은 1[π > 0.5]·μ_W다. controller로 가는 W̄_{t+1}, W̄_{t+2}는 μ_W이고, 측정 힘 \|W̄_t\|가 1 N 미만이면 0이다(`wrench_gate`). |
| controller | aligned, κ = 1, full Λ(아래). |

학습 명령(z_dim 16 기준; noise 값은 기본값이라 따로 주지 않는다):

```bash
lerobot-train --policy.type=latent_sde --policy.z_dim=16 '--policy.down_dims=[512,1024,2048]' \
  --policy.use_ft=true --policy.impedance_targets=true --policy.push_to_hub=false \
  --env.type=ft4 --env.impedance=true \
  --dataset.repo_id=himchan00/ft4v2_reactive --dataset.root=<ft4v2_reactive/policy> --dataset.video_backend=torchcodec \
  --batch_size=64 --steps=30000 --save_freq=5000 --env_eval_freq=10000 \
  --eval.n_episodes=50 --eval.batch_size=10 --eval.use_async_envs=false \
  --output_dir=outputs/train/<run> --job_name=<run>
```

**Controller 비교** (reactive z16, noise 없음, wrench gate 1 N, /240)

비교한 변형:
- reference 시작: "이어 붙이기"는 tick마다 reference를 직전 tick reference의 끝 상태에서 이어 간다(처음 구현). "aligned"는 tick마다 측정 상태(pose, 속도, 직전 tick의 평균 가속도)에서 새로 시작하고, F_d도 측정 W̄_t에서 시작한다.
- κ: 정책의 목표(현재 pose 기준 상대값)를 어디에 붙이는지 정한다. κ = 1이면 측정 pose에, κ = 0.5면 측정 pose와 직전 reference 끝의 중간에 붙인다.
- z: prior에서 샘플하거나, prior 평균을 쓴다.
- Λ: full은 위치와 자세를 묶은 6×6 operational-space 관성이다. decoupled는 robosuite OSC 기본처럼 둘을 따로 계산한다.

| reference | κ | z | full Λ | decoupled Λ |
|---|---|---|---|---|
| 이어 붙이기 | 1 | 샘플 | 50 | 31 |
| 이어 붙이기 | 1 | 평균 | 69 | 30 |
| 이어 붙이기 | 0.5 | 샘플 | 112 | 35 |
| 이어 붙이기 | 0.5 | 평균 | 138 | 46 |
| **aligned (채택)** | **1** | **샘플** | **81** | 47 |
| aligned | 1 | 평균 | 103 | 60 |

- **full Λ.** decoupled는 모든 조합에서 크게 떨어지고 Square와 Door가 0에 가까워진다.
  - full Λ는 예측한 다음 pose x̂_{t+1}을 0.1–0.4 mm로 따라가지만, decoupled는 0.6–2.0 mm(Door 접촉 중 3.5 mm) 벗어난다.
  - impedance label은 데모에서 실제로 일어난 결과라 controller가 그대로 재현해야 한다. OSC label은 데모 때 같은 controller에 준 명령이라 decoupled여도 괜찮다.
- **aligned.** 같은 κ = 1, 같은 z에서 이어 붙이기보다 항상 낫다(full Λ에서 +31, +34).
- **κ = 1, z 샘플.**
  - 점수는 이어 붙이기 + κ 0.5 + z 평균이 가장 높다.
  - κ 0.5는 직전 reference의 끝을 목표에 섞는 별도의 적분기다. aligned는 직전 reference를 쓰지 않으므로 κ = 1을 짝으로 정했다. aligned + κ 0.5는 재지 않았다.
  - z는 정책 정의대로 샘플한다.
- **wrench gate 1 N.** 켰을 때와 껐을 때의 결과가 엇갈린다.
  - 이어 붙이기 + κ 0.5 + z 평균: 138 vs 128(켬이 낫다).
  - 이어 붙이기 + κ 1 + z 샘플: 50 vs 62(끔이 낫다).
  - aligned에서는 비교하지 않았다.
  - sim에서는 free space의 W̄가 정확히 0이라 1 N 문턱이 의미가 있다. 실제 센서에서는 다시 정해야 한다.

**Wrench noise.** state noise처럼 F/T 입력에도 교정용 noise를 준다.
- 정의:
  - 학습 때 drift 입력을 W̃_k = W̄_k + ε_k로 바꾼다. ε_k ~ N(0, (σ‖W̄_k‖)²)은 힘과 토크 3-vector마다 따로 뽑으므로, free space에서는 0이다.
  - label W̄_{t+1}은 (1−β)ε_k만큼 옮긴다. 다음 tick에 noise의 β만큼 되돌아오라는 뜻이다.
  - 값은 σ = 0.3, β = 0.3(`wrench_noise_std`, `wrench_noise_gain`)이다.
  - σ 0.3은 W̄가 한 tick에 변하는 양(중앙값)의 4–8배이고, Wipe에서는 약 4.2 N이다.
- state noise(sn): pose 입력에 같은 방식으로 noise를 주고 label을 다시 만든다. σ = 1(action scale 단위, 축당 2.27 mm / 0.51°), β = 0.3이다.

같은 평가 방법, 같은 controller(위 고정 세팅), z16:

| run | Square | Stack | Wipe | Door | 합계 /240 | seed별 /80 |
|---|---|---|---|---|---|---|
| noise 없음 (`..._onestep_constant_z16_30k`) | 15 | 39 | 3 | 21 | 78 (32.5%) | 31 / 25 / 22 |
| wn (`..._z16_wn03_g03_30k`) | 19 | 22 | 5 | 59 | 105 (43.8%) | 41 / 32 / 32 |
| wn + sn (`..._z16_wn03_g03_sn1_g03_30k`) | 40 | 18 | 0 | 59 | 117 (48.8%) | 42 / 38 / 37 |

(run 이름 앞부분은 `latent_sde_ft4v2_reactive_ft_tcp_onestep_constant_`이다.)

- 합계는 noise 없음 대비 wn이 z ≈ 2.5, wn + sn이 z ≈ 3.6으로 유의하게 오른다. 세 seed 모두 같은 방향이다. wn과 wn + sn의 차이는 유의하지 않다(z ≈ 1.1).
- 이득의 대부분은 Door다(21 → 59).
- Square는 state noise를 같이 넣을 때만 오른다(15 → 40).
- Stack은 39 → 18–22로 유의하게 떨어진다(z ≈ 3.1, 3.8). 원인은 아직 보지 않았다.
- Wipe는 여전히 거의 0이다.
- 해석(검증하지 않음): noise 전 Door 실패를 분석해 보니, 정책이 측정 힘을 "문을 얼마나 돌렸나"의 신호로 쓰다가 너무 일찍 당기고 있었다.
  - F/T 입력만 1.3배로 키우면 Door가 43 → 4/60으로 무너졌다(native 체크포인트, 이전 controller). 입력 힘↑ → F_d↑ → 측정 힘↑의 양의 되먹임이다.
  - W̄ 입력에 noise를 주면 힘의 정확한 크기에 덜 기대게 되어, 이 되먹임이 약해졌을 수 있다.

학습 중 eval(30k, 과제당 50 에피소드, %):

| run | Square | Stack | Wipe | Door | 전체 |
|---|---|---|---|---|---|
| z16 wn | 30 | 60 | 20 | 96 | 51.5 |
| z16 wn + sn | 52 | 36 | 0 | 98 | 46.5 |
| z32 wn | 42 | 66 | 16 | 100 | 56.0 |
| z32 wn + sn | 50 | 44 | 0 | 100 | 48.5 |

- z_dim 32는 16과 같은 수준이다(+2–5, 오차 범위).
- sn이 Square를 올리고 Stack·Wipe를 내리는 경향은 두 z_dim에서 같다.
- 여기서는 wn이 wn + sn보다 높아 위 /240 표와 순서가 반대다. 그래서 둘 중 무엇이 나은지는 정하지 않았다. 기본값은 진행 중인 run과 같은 wn이다.

예전 controller에서 state noise만 넣었을 때(학습 중 eval):
- native 데이터 impedance: 12.0 → 23.5%(Square 8 → 58, Stack 28 → 16).
- OSC policy에 같은 크기의 state noise: 이득이 없었다(50.5 → 46.5%).

**진행 중**(2026-10-27 시작) — z32 wn 세팅에서 한 가지씩 바꾼 두 run:
- `latent_sde_ft4v2_native_ft_tcp_onestep_constant_z32_wn03_g03_30k`: reactive 대신 native 데이터.
- `latent_sde_ft4v2_reactive_ft_tcp_osc_z32_wn03_g03_30k`: impedance 대신 OSC 출력(`--policy.impedance_targets`, `--env.impedance`를 뺌). OSC도 같은 W̄_k 입력과 `ft_scale`, 같은 wrench noise를 쓴다(label 이동은 없음). 그래서 F/T 입력까지 맞춘 OSC 비교가 된다.

**남은 문제: Wipe.** eval 노트북(`src/lerobot/policies/latent_sde/ft4_eval_viz.ipynb`)으로 wn + sn의 seed 1000 에피소드를 보면, 접촉 tick의 85–91%가 "uneven"(압력 중심이 가운데 60% 밖)이다.
- 압력 중심은 패드 긴 축의 가장자리에 붙어 있다.
- 패드는 접촉 전부터 짧은 축을 중심으로 3–5° 기울어 있다.
- 마커 100개 중 84–88개는 패드가 한 번도 지나가지 않았다.

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
