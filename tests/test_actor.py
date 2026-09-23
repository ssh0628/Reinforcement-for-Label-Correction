"""Regression tests for dataset-level Actor gradient accumulation."""

import unittest

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import SGD

from rl.actor import select_actor_queries, update_actor
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
        self.hidden = nn.Linear(4, 5)
        self.normalization = nn.BatchNorm1d(5)
        self.projection = nn.Linear(5, 3)
        with torch.no_grad():
            self.normalization.running_mean.copy_(torch.linspace(-0.4, 0.4, 5))
            self.normalization.running_var.copy_(torch.linspace(0.7, 1.3, 5))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.projection(torch.tanh(self.normalization(self.hidden(inputs))))


def identity_preprocess(
    inputs: Tensor, device: torch.device, mean: Tensor, std: Tensor
) -> Tensor:
    del mean, std
    return inputs.to(device=device, dtype=torch.float32)


def encode(model: nn.Module, inputs: Tensor) -> Tensor:
    return model(inputs)


class ActorGradientAccumulationTest(unittest.TestCase):
    def test_query_selection_is_reproducible_and_step_specific(self) -> None:
        first = select_actor_queries(20, 5, seed=3, step=7)
        repeated = select_actor_queries(20, 5, seed=3, step=7)
        next_step = select_actor_queries(20, 5, seed=3, step=8)

        torch.testing.assert_close(first, repeated)
        self.assertEqual(torch.unique(first).numel(), 5)
        self.assertFalse(torch.equal(first, next_step))

    def _assert_matches_direct_update(
        self,
        *,
        microbatch_size: int,
        query_weights: Tensor,
        queries: Tensor | None = None,
    ) -> None:
        torch.manual_seed(7)
        sample_count = 8
        raw_images = torch.randn(sample_count, 4)
        with torch.inference_mode():
            inference_labels = F.one_hot(torch.arange(sample_count) % 3, num_classes=3).float()
            inference_neighbors = torch.tensor(
                [
                    [3, 1, 1], [4, 0, 0], [5, 1, 1], [0, 1, 1],
                    [1, 0, 0], [2, 1, 1], [0, 1, 1], [1, 0, 0],
                ]
            )
            inference_actions = torch.tensor(
                [True, False, True, True, False, True, False, True]
            )
        labels = inference_labels.clone()
        neighbors = inference_neighbors.clone()
        actions = inference_actions.clone()
        q_value = torch.tensor(1.7, requires_grad=True)
        policy = LabelCorrectionPolicy(temperature=0.9, correction_chunk_size=sample_count)
        selected_queries = torch.arange(sample_count) if queries is None else queries
        selected_neighbors = neighbors[selected_queries]

        reference = TinyBackbone().eval()
        accumulated = TinyBackbone()
        accumulated.load_state_dict(reference.state_dict())
        initial_parameters = {
            name: parameter.detach().clone() for name, parameter in reference.named_parameters()
        }
        initial_buffers = {name: buffer.clone() for name, buffer in reference.named_buffers()}
        reference_optimizer = SGD(reference.parameters(), lr=0.05)
        accumulated_optimizer = CountingSGD(accumulated.parameters(), lr=0.05)

        reference_embeddings = reference(raw_images)
        reference_embeddings.retain_grad()
        reference_step = policy(
            reference_embeddings[selected_queries],
            reference_embeddings[selected_neighbors],
            labels[selected_queries],
            labels[selected_neighbors],
            actions=actions[selected_queries],
        )
        # One complete autograd graph; explicit query weights define the objective.
        reference_loss = -q_value.detach() * (
            reference_step.log_probabilities * query_weights
        ).sum()
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss.backward()
        reference_optimizer.step()

        accumulated.eval()
        with torch.inference_mode():
            detached_embeddings = accumulated(raw_images).float()
        # update_actor must freeze BatchNorm before re-encoding even from training mode.
        accumulated.train()
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
            microbatch_size=microbatch_size,
            query_indices=queries,
            use_amp=False,
            amp_dtype=torch.float32,
            preprocess=identity_preprocess,
            encode=encode,
        )

        torch.testing.assert_close(
            torch.tensor(accumulated_loss), reference_loss.detach(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(accumulated_optimizer.step_count, 1)
        self.assertIsNone(q_value.grad)
        self.assertFalse(accumulated.training)
        self.assertFalse(accumulated.normalization.training)
        for name, actual in accumulated.named_buffers():
            torch.testing.assert_close(actual, initial_buffers[name], rtol=0, atol=0)
        for (name, expected), (actual_name, actual) in zip(
            reference.named_parameters(), accumulated.named_parameters(), strict=True
        ):
            self.assertEqual(actual_name, name)
            self.assertIsNotNone(expected.grad)
            self.assertIsNotNone(actual.grad)
            self.assertGreater(float(expected.grad.norm()), 1e-6, msg=name)
            self.assertGreater(
                float((actual.detach() - initial_parameters[name]).norm()), 1e-7, msg=name
            )
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=1e-6)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        if queries is not None:
            # Image 1 participates only as a repeated/shared neighbor, never as a query.
            self.assertNotIn(1, queries.tolist())
            self.assertGreater(float(reference_embeddings.grad[1].norm()), 1e-6)
            inactive_indices = torch.tensor([2, 4, 5, 7])
            torch.testing.assert_close(
                reference_embeddings.grad[inactive_indices], torch.zeros(4, 3), rtol=0, atol=0
            )

    def test_uneven_microbatches_match_sum_of_batch_means(self) -> None:
        self._assert_matches_direct_update(
            microbatch_size=3,
            query_weights=torch.tensor([1 / 3] * 6 + [1 / 2] * 2),
        )

    def test_selected_queries_match_sum_of_batch_means(self) -> None:
        self._assert_matches_direct_update(
            microbatch_size=2,
            queries=torch.tensor([0, 3, 6]),
            query_weights=torch.tensor([1 / 2, 1 / 2, 1]),
        )

    def test_selected_queries_use_actual_query_count(self) -> None:
        self._assert_matches_direct_update(
            microbatch_size=16,
            queries=torch.tensor([0, 3, 6]),
            query_weights=torch.tensor([1 / 3] * 3),
        )

    def test_equal_microbatches_scale_full_mean_by_batch_count(self) -> None:
        self._assert_matches_direct_update(
            microbatch_size=2,
            query_weights=torch.full((8,), 4 / 8),
        )


if __name__ == "__main__":
    unittest.main()
