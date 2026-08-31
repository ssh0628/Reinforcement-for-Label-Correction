from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from rl.knn import build_exact_clean_knn, build_exact_policy_knn


class ExactKNNTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.embeddings = torch.randn(13, 5, generator=generator)

    def test_policy_knn_matches_direct_distance_search(self) -> None:
        neighbors = build_exact_policy_knn(
            self.embeddings,
            k=4,
            query_chunk_size=3,
            reference_chunk_size=5,
        )
        distances = torch.cdist(self.embeddings, self.embeddings).square()
        distances.fill_diagonal_(float("inf"))
        expected = distances.topk(4, dim=1, largest=False, sorted=True).indices

        self.assertTrue(torch.equal(neighbors, expected))

    def test_clean_knn_matches_direct_distance_search(self) -> None:
        actions = torch.tensor(
            [True, False, False, True, False, True, False, False, True, False, False, False, True]
        )
        result = build_exact_clean_knn(
            self.embeddings,
            actions,
            k=3,
            query_chunk_size=2,
            reference_chunk_size=4,
        )
        noisy_indices = actions.nonzero().flatten()
        clean_indices = (~actions).nonzero().flatten()
        distances = torch.cdist(
            self.embeddings[noisy_indices], self.embeddings[clean_indices]
        ).square()
        expected_positions = distances.topk(3, dim=1, largest=False, sorted=True).indices
        expected_neighbors = clean_indices[expected_positions]
        normalized = F.normalize(self.embeddings, dim=1)
        expected_cosines = (
            normalized[noisy_indices].unsqueeze(1) * normalized[expected_neighbors]
        ).sum(dim=2)

        self.assertTrue(torch.equal(result.noisy_indices, noisy_indices))
        self.assertTrue(torch.equal(result.neighbor_indices, expected_neighbors))
        self.assertTrue(torch.allclose(result.neighbor_cosine_similarities, expected_cosines))


if __name__ == "__main__":
    unittest.main()
