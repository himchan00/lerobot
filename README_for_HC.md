# HC 개인용 환경 메모

LeRobot 레포에서 Push-T 벤치마크 + Diffusion Policy 학습용 셋업 기록.

## 1. 추가 설치 (base install 끝난 뒤)

레포 루트(`/home/v-hihwang/lerobot`)에서:

```bash
pip install -e '.[pusht,diffusion,latent_ode,training]'
```

각 extra가 가져오는 핵심 패키지:

| extra | 패키지 | 역할 |
|---|---|---|
| `pusht` | `gym-pusht`, `pymunk` | Push-T 시뮬레이션 환경 |
| `diffusion` | `diffusers` | Diffusion Policy의 noise scheduler / UNet 빌딩 블록 |
| `latent_ode` | `vector-quantize-pytorch` | Latent SDE Policy의 VQ 코드북 (`use_vq=True`일 때 필요) |
| `training` | `accelerate`, `wandb` | 학습 루프 |

> `latent_ode` extra를 빠뜨리면 `use_vq=True`인 `latent_ode` 정책 학습 시 `ModuleNotFoundError: No module named 'vector_quantize_pytorch'`가 발생함. (Diffusion Policy만 돌릴 거면 생략 가능.)

설치 확인:

```bash
python -c "import gym_pusht, pymunk, diffusers, vector_quantize_pytorch, accelerate, wandb; print('ok')"
```

## 2. 학습 커맨드 (Push-T + Diffusion Policy)

```bash
lerobot-train \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/pusht \
    --env.type=pusht \
    --output_dir=outputs/train/diffusion_pusht \
    --job_name=diffusion_pusht \
    --batch_size=64 \
    --eval.use_async_envs=false \
    --wandb.enable=True \
    --wandb.project=lerobot_pusht
```

> `batch_size=64`: `examples/training/train_policy.py`의 Diffusion+PushT 예제 값을 사용.

> `--eval.use_async_envs=false`: Push-T는 eval async forkserver 워커에서 gym_pusht가 재등록 안 돼서 NamespaceNotFound 발생해서 -> false로 지정해서 우회.

콘솔 스크립트가 안 잡히면 `python -m lerobot.scripts.lerobot_train ...` 로도 가능.

## 2.1 Eval 커맨드 (저장된 체크포인트 필요)

```bash
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/100000/pretrained_model \
    --env.type=pusht \
    --eval.n_episodes=50 \
    --eval.use_async_envs=false \
    --output_dir=outputs/eval/diffusion_pusht \
    --job_name=diffusion_pusht_eval \
    --seed=1000
```
- `--policy.*` 오버라이드로 학습 시 config의 정책 옵션을 평가용으로만 바꿀 수 있음 (`deterministic_inference`, `n_action_steps` 등).


## 3. 메모

- **Robot class 지정 불필요/불가**: Push-T는 sim 벤치마크라 `TrainPipelineConfig`에 `robot` 필드 자체가 없음. `--robot.type=...`는 `lerobot-record` 같은 실제 하드웨어용 스크립트에서만 사용.
- Diffusion Policy의 디폴트 하이퍼파라미터(`src/lerobot/policies/diffusion/configuration_diffusion.py`)는 이미 Push-T 기준으로 튜닝되어 있어 추가 인자 거의 불필요.
- 학습 산출물(체크포인트/eval 비디오): `outputs/train/diffusion_pusht/`.
- 데이터셋 `lerobot/pusht`는 HF Hub에서 첫 실행 시 자동 다운로드 (`~/.cache/huggingface/`).
- 파이썬 스크립트 형태 예시는 `examples/training/train_policy.py` (이 파일 자체가 Diffusion + Push-T 조합).

## 4. AMLT (Singularity) 셋업 시 주의사항

`amlt/diffusion_pusht.yaml`로 H100 클러스터(`msrresrchbasicvc`)에 잡 던질 때 밟은 함정들:

- **HF 캐시 분리**: `/mnt/v-hihwang`은 rslex 마운트라 `statvfs`가 항상 0을 반환 → `datasets` 라이브러리의 disk-space 체크에 막혀 `OSError: Not enough disk space` 발생. 해결: persistent 캐시는 `/mnt`에, builder 임시 캐시만 로컬 `/scratch`에 분리.
  ```yaml
  HF_HOME: /mnt/v-hihwang/projects/lerobot/hf_cache       # 큰 데이터셋 본체 (persistent)
  HF_HUB_CACHE: /mnt/v-hihwang/projects/lerobot/hf_cache/hub
  HF_DATASETS_CACHE: /scratch/hf_datasets_cache           # 작은 builder 임시 (재생성 OK)
  ```
- **FIPS abort 회피**: H100 노드는 FIPS enforce. pip wheel `av` (PyAV)가 번들한 OpenSSL이 import 시 `OpenSSL internal error: FATAL FIPS SELFTEST FAILURE`로 즉시 abort. 해결: setup에 두 줄 추가.
  ```yaml
  - conda run --name lerobot pip uninstall -y av || true
  - conda run --name lerobot conda install -y -c conda-forge av
  ```
- **Preemption resume**: 출력 디렉터리는 `/mnt`에 두고 `${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json` 존재 여부로 분기 → 있으면 `--resume=true --config_path=$CKPT_CFG`, 없으면 fresh start (stale dir은 `rm -rf`로 정리해야 lerobot의 `FileExistsError` 회피).
- **`--eval.use_async_envs=false`**는 AMLT에서도 그대로 필요 (forkserver 워커가 `gym_pusht` import 안 함).
