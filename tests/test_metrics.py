from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from evaluate.metrics import evaluate_classifier


def _identity_preprocess(
    images: Tensor,
    device: torch.device,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    del mean, std
    return images.to(device)


class ClassifierEvaluationTest(unittest.TestCase):
    def test_chunked_evaluation_matches_direct_metrics(self) -> None:
        model = nn.Linear(3, 2, bias=False)
        model.weight.data.copy_(torch.tensor([[0.5, -0.2, 0.1], [-0.3, 0.4, 0.2]]))
        images = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )
        labels = torch.tensor([0, 1, 1, 0, 1])
        logits = model(images)
        expected_loss = nn.functional.cross_entropy(logits, labels)
        expected_accuracy = logits.argmax(dim=1).eq(labels).float().mean()

        summary = evaluate_classifier(
            model,
            images,
            labels,
            torch.device("cpu"),
            torch.empty(0),
            torch.empty(0),
            batch_size=2,
            use_amp=False,
            amp_dtype=torch.float32,
            preprocess=_identity_preprocess,
        )

        self.assertAlmostEqual(summary["loss"], float(expected_loss.detach()), places=6)
        self.assertAlmostEqual(summary["accuracy"], float(expected_accuracy), places=6)


if __name__ == "__main__":
    unittest.main()
