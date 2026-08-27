"""Regression tests for dataset-level Actor gradient accumulation."""

import unittest

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD

from rl.actor import update_actor
from rl.policy import LabelCorrectionPolicy


class CountingSGD(SGD):
    def __init__(self, parameters, *, lr: float) -> None:
        super().__init__(parameters, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 3, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(inputs)


def identity_preprocess(
    inputs: Tensor, device: torch.device, mean: Tensor, std: Tensor
) -> Tensor:
    del mean, std
    return inputs.to(device=device, dtype=torch.float32)


def encode(model: nn.Module, inputs: Tensor) -> Tensor:
    return model(inputs)


class ActorGradientAccumulationTest(unittest.TestCase):
    def test_microbatches_match_one_full_dataset_update(self) -> None:
        torch.manual_seed(7)
        sample_count = 8
        raw_images = torch.randn(sample_count, 4)
        with torch.inference_mode():
            inference_labels = F.one_hot(torch.arange(sample_count) % 3, num_classes=3).float()
            inference_neighbors = torch.tensor(
                [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [0, 1]]
            )
            inference_actions = torch.tensor(
                [True, False, True, True, False, True, False, True]
            )
        labels = inference_labels.clone()
        neighbors = inference_neighbors.clone()
        actions = inference_actions.clone()
        q_value = torch.tensor(1.7)
        policy = LabelCorrectionPolicy(temperature=0.5, correction_chunk_size=sample_count)

        reference = TinyBackbone()
        accumulated = TinyBackbone()
        accumulated.load_state_dict(reference.state_dict())
        reference_optimizer = SGD(reference.parameters(), lr=0.05)
        accumulated_optimizer = CountingSGD(accumulated.parameters(), lr=0.05)

        reference_embeddings = reference(raw_images)
        reference_step = policy(
            reference_embeddings,
            reference_embeddings[neighbors],
            labels,
            labels[neighbors],
            actions=actions,
        )
        reference_loss = -q_value * reference_step.log_probabilities.mean()
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss.backward()
        reference_optimizer.step()

        with torch.inference_mode():
            detached_embeddings = accumulated(raw_images).float()
        accumulated_loss = update_actor(
            accumulated,
            policy,
            accumulated_optimizer,
            torch.amp.GradScaler("cuda", enabled=False),
            raw_images,
            inference_labels,
            detached_embeddings,
            inference_neighbors,
            inference_actions,
            q_value,
            torch.device("cpu"),
            torch.empty(0),
            torch.empty(0),
            microbatch_size=3,
            use_amp=False,
            amp_dtype=torch.float32,
            preprocess=identity_preprocess,
            encode=encode,
        )

        self.assertAlmostEqual(accumulated_loss, float(reference_loss.detach()), places=6)
        self.assertEqual(accumulated_optimizer.step_count, 1)
        for expected, actual in zip(reference.parameters(), accumulated.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
