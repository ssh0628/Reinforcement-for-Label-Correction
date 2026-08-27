# Reinforcement Learning for Noisy Label Correction

CIFAR-10 RLNLC experiment code. The active implementation lives directly in
this repository; the previous implementation is preserved under `_past/`.

## Pipeline

Run each stage from the repository root after editing `setting/config.py`:

```bash
python cifar_noise.py
python cifar_warmup.py
python cifar_rl.py
python cifar_correction.py
python cifar_finetuning.py
python cifar_evaluate.py
```

`knn_correction.py` provides the weighted-KNN baseline and
`cifar_knn_quality.py` evaluates the warm-up feature space.

The RL environment always computes features, KNN, label correction, reward,
and Q over the configured training set. The Actor then samples one random
mini-batch and performs exactly one optimizer update per trajectory step.
