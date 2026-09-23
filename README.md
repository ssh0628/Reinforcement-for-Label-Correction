# RLNLC

- 논문 재현 대상: [Learning to Clean: Reinforcement Learning for Noisy Label Correction](https://arxiv.org/abs/2511.19808)
- 데이터셋: CIFAR-10
- 목적: noisy label 탐지·보정 및 보정 라벨 기반 분류기 학습
- 구성: KNN policy + Actor–Critic + iterative label correction
- 지원 노이즈: Symmetric Noise(SN), Instance-Dependent Noise(IDN)
- 지원 backbone: CIFAR ResNet-18, CIFAR ResNet-34
- 학습 데이터: CIFAR-10 train 50,000장

## Pipeline

```text
Noise generation
  → Warm-up
  → Reinforcement learning
  → Label correction
  → Fine-tuning
  → Evaluation
```

| 단계 | 실행 파일 | 주요 결과 |
|---|---|---|
| Noise | `cifar_noise.py` | noisy label, noise mask |
| Warm-up | `cifar_warmup.py` | warm-up backbone checkpoint |
| RL | `cifar_rl.py` | Actor/Critic checkpoint, RL log |
| Correction | `cifar_correction.py` | corrected soft labels |
| Fine-tuning | `cifar_finetuning.py` | fine-tuned classifier checkpoint |
| Evaluation | `cifar_evaluate.py` | CIFAR-10 test accuracy/loss |

## Project Structure

### Entry Points

| 파일 | 역할 |
|---|---|
| `run.py` | Warm-up부터 Evaluation까지 전체 파이프라인 실행 |
| `cifar_noise.py` | CIFAR-10 synthetic noisy-label artifact 생성 |
| `cifar_warmup.py` | noisy label 기반 backbone 사전 학습 |
| `cifar_rl.py` | Actor–Critic 강화학습 실행 |
| `cifar_correction.py` | 학습된 Actor를 이용한 반복 라벨 보정 |
| `cifar_finetuning.py` | 보정된 soft label 기반 classifier fine-tuning |
| `cifar_evaluate.py` | 선택된 fine-tuning checkpoint 최종 평가 |

### `setting/`

| 파일 | 역할 |
|---|---|
| `config.py` | 경로, 모델, 노이즈, 학습, checkpoint, AMP 설정 |
| `data.py` | CIFAR-10 loader, transform, preprocessing, model 생성 |
| `model.py` | CIFAR용 ResNet-18/34 및 feature extraction interface |
| `noise.py` | SN/IDN 생성 및 artifact 저장 |
| `warmup.py` | warm-up 학습, validation, checkpoint 저장 |

### `rl/`

| 파일 | 역할 |
|---|---|
| `run.py` | RL stage 진입점 및 결과 경로 연결 |
| `engine.py` | trajectory 생성, Actor/Critic 학습, 검증, checkpoint 관리 |
| `actor.py` | action sampling, Actor loss, embedding-gradient 기반 업데이트 |
| `critic.py` | consistency histogram encoding, Critic MLP, TD update |
| `policy.py` | KNN 기반 correction probability 및 action 처리 |
| `reward.py` | Label Consistency Reward와 Noisy Label Alignment Reward 계산 |
| `knn.py` | chunk 기반 exact KNN 탐색 |

### `evaluate/`

| 파일 | 역할 |
|---|---|
| `correction.py` | Actor checkpoint 로드 및 corrected-label artifact 생성 |
| `finetuning.py` | corrected soft label 기반 classifier 재학습 |
| `final.py` | CIFAR-10 test 10,000장 최종 평가 |
| `metrics.py` | correction accuracy, precision, recovery, preservation 지표 계산 |

### `log/` 및 `tests/`

| 파일 | 역할 |
|---|---|
| `log/common.py` | 실행 로그, CSV, timing 공통 기능 |
| `log/rl.py` | RL 로그 파일명과 CSV schema |
| `tests/test_actor.py` | Actor query 선택과 gradient update 회귀 테스트 |
| `tests/test_config.py` | 주기적 RL checkpoint 설정 테스트 |
| `tests/test_knn.py` | exact KNN 연산 테스트 |
| `tests/test_metrics.py` | 평가 지표 계산 테스트 |

## Setup

- 권장 환경: Python 3.12+, CUDA GPU
- 의존성 설치:

```bash
uv sync
```

- 실험 설정: `setting/config.py`
- 데이터 경로: `cifar10/`
- 출력 경로: `cifar_output/<experiment_name>/`

## Run

### 1. Noise artifact 생성

```bash
uv run python cifar_noise.py
```

### 2. 전체 파이프라인 실행

```bash
uv run python run.py
```

### 3. 특정 단계부터 재개

```bash
uv run python run.py --start-from cifar_rl.py
```

### 4. 단계별 실행

```bash
uv run python cifar_warmup.py
uv run python cifar_rl.py
uv run python cifar_correction.py
uv run python cifar_finetuning.py
uv run python cifar_evaluate.py
```

## Outputs

| 출력 | 내용 |
|---|---|
| `warmup/model/` | warm-up checkpoint |
| `model_rl/` | Actor/Critic checkpoint, corrected soft labels |
| `model_finetune/` | fine-tuning checkpoint |
| `logs/<stage>/run.log` | 실행 설정 및 단계별 로그 |
| `logs/<stage>/train.csv` | epoch/step 학습 지표 |
| `logs/<stage>/run_summary.csv` | 실행 결과 요약 |
| `logs/<stage>/timing.csv` | 단계별 실행 시간 |

- 기존 결과: 기본적으로 덮어쓰기 방지
- 재실행: `setting/config.py`의 해당 `overwrite_*` 옵션 사용
- 로컬 실험 자료 및 진단 코드: `experiment_folder/`
- `experiment_folder/`, `cifar10/`, `cifar_output/`: Git 제외

## Tests

```bash
uv run python -m unittest discover -s tests
```
