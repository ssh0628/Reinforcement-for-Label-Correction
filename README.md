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

The RL environment computes features, KNN, label correction, reward, Q, and
the Actor policy loss over the entire configured training set. Actor gradients
are accumulated in memory-bounded microbatches, followed by exactly one
optimizer update per trajectory step.
