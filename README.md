# RLNLC

- CIFAR-10 noisy-label correction 실험 코드
- Symmetric Noise / Instance-Dependent Noise 지원
- CIFAR ResNet-18 / ResNet-34 지원
- Warm-up → KNN 분석 → RL correction → Fine-tuning → Evaluation

## 환경

- Python 3.12+
- PyTorch 2.4+
- CUDA GPU 권장

```bash
uv sync
```

## 설정

- 설정 파일: `setting/config.py`
- 주요 항목
  - 데이터 수, seed, noise 종류·비율
  - backbone, augmentation
  - warm-up checkpoint: `best` / `last`
  - RL epoch, Actor query 수, terminal target, remaining horizon
  - Actor·Critic optimizer와 learning rate
  - correction step, fine-tuning checkpoint
  - experiment name, overwrite 옵션
- 서버 경로에 맞게 `PROJECT_ROOT` 수정

## 실행

```bash
# 최초 1회 또는 noise 설정 변경 시
uv run python cifar_noise.py

# Warm-up부터 Evaluation까지
uv run python run.py
```

중간 단계부터 재개:

```bash
uv run python run.py --start-from cifar_rl.py
```

개별 실행 순서:

```bash
uv run python cifar_warmup.py
uv run python cifar_knn_quality.py
uv run python cifar_rl.py
uv run python cifar_correction.py
uv run python cifar_finetuning.py
uv run python cifar_evaluate.py
```

KNN baseline:

```bash
uv run python knn_correction.py
```

## Actor update

- 샘플별 Bernoulli action으로 joint action 구성
- Actor loss

  ```text
  L_actor = -Q(s, a) × mean(log π(a_i | s))
  ```

- 선택 query를 microbatch로 나누어 gradient 누적
- RL trajectory step당 `optimizer.step()` 1회
- microbatch 분할은 gradient 평균과 결과에 영향 없음

## 출력

- 기본 경로: `cifar_output/<experiment_name>/`
- 모델: warm-up, Actor, Critic, fine-tuning checkpoint
- 로그: `run.log`, `train.csv`, `run_summary.csv`, `timing.csv`
- corrected labels: `train_corrected_soft_labels.npy`
- 기존 결과 보호: 해당 `runtime.overwrite_*`가 `False`이면 덮어쓰기 중단
