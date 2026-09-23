# RLNLC

CIFAR-10의 noisy label을 KNN policy와 Actor–Critic 학습으로 보정하는 연구 구현입니다.
실험용 진단·ablation 코드는 메인 파이프라인과 분리하고, 재현에 필요한 핵심 단계만 유지합니다.

## Pipeline

```text
Noise generation
  → Warm-up
  → Reinforcement learning
  → Label correction
  → Fine-tuning
  → Evaluation
```

- Noise: Symmetric Noise 또는 Instance-Dependent Noise
- Backbone: CIFAR ResNet-18 / ResNet-34
- Policy: feature-space KNN 기반 label correction
- Critic: label-consistency histogram을 입력으로 사용하는 MLP
- Data: CIFAR-10 train 50,000장 전체 사용

## Structure

```text
evaluate/   correction, fine-tuning, final evaluation
log/        CSV 및 실행 로그 유틸리티
rl/         actor, critic, policy, reward, KNN, training engine
setting/    configuration, data, noise, model, warm-up
tests/      핵심 연산 회귀 테스트
run.py      전체 파이프라인 실행기
```

로컬 실험 결과와 진단 스크립트는 `experiment_folder/`에 보관하며 Git에서 제외합니다.

## Setup

Python 3.12 이상과 CUDA GPU 사용을 권장합니다.

```bash
uv sync
```

설정은 `setting/config.py`에서 관리합니다. 데이터와 출력 경로는 저장소 루트를 기준으로 생성되며,
모델·노이즈·학습률·checkpoint·AMP 설정을 한 곳에서 조정할 수 있습니다.

## Run

노이즈 artifact를 먼저 생성합니다.

```bash
uv run python cifar_noise.py
```

전체 파이프라인을 실행합니다.

```bash
uv run python run.py
```

특정 단계부터 재개할 수 있습니다.

```bash
uv run python run.py --start-from cifar_rl.py
```

각 단계는 독립적으로도 실행할 수 있습니다.

```bash
uv run python cifar_warmup.py
uv run python cifar_rl.py
uv run python cifar_correction.py
uv run python cifar_finetuning.py
uv run python cifar_evaluate.py
```

## Actor update

한 RL step에서 선택된 query의 Bernoulli action을 샘플링하고 다음 loss를 사용합니다.

```text
L_actor = Σ_B [-Q(s, a) × mean_{i ∈ B}(log π(a_i | s))]
```

전체 feature graph를 보관하지 않기 위해 embedding gradient를 캐시합니다.

```text
z = f_θ(x),  g = ∂L_actor/∂z
surrogate = Σ(z × stop_gradient(g))
```

microbatch별 `backward()`는 gradient만 누적하며, Actor 파라미터는 RL step당 한 번 업데이트합니다.

## Outputs

실행 결과는 `cifar_output/<experiment_name>/`에 저장됩니다.

- warm-up, Actor, Critic, fine-tuning checkpoints
- corrected soft labels
- `run.log`, `train.csv`, `run_summary.csv`, `timing.csv`

기존 출력이 있을 때는 기본적으로 덮어쓰지 않습니다. 의도적으로 다시 생성할 때만
`setting/config.py`의 해당 `overwrite_*` 옵션을 활성화합니다.

## Tests

```bash
uv run python -m unittest discover -s tests
```
