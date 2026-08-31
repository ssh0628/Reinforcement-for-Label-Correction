"""Regression tests for periodic RL checkpoint paths."""

import unittest
from dataclasses import replace

from setting.config import CIFARConfig


class PeriodicCheckpointConfigTest(unittest.TestCase):
    def test_interval_is_independent_of_total_epochs(self) -> None:
        base = CIFARConfig()
        config = replace(base, rl=replace(base.rl, epochs=120, checkpoint_interval=50))

        self.assertEqual(config.rl_checkpoint_epochs, (50, 100))
        self.assertEqual(
            [path.name for path in config.rl_periodic_checkpoint_paths],
            [
                "actor_epoch_0050.pt",
                "critic_epoch_0050.pt",
                "actor_epoch_0100.pt",
                "critic_epoch_0100.pt",
            ],
        )

    def test_500_epochs_produce_ten_periodic_pairs(self) -> None:
        base = CIFARConfig()
        config = replace(base, rl=replace(base.rl, epochs=500, checkpoint_interval=50))

        self.assertEqual(config.rl_checkpoint_epochs, tuple(range(50, 501, 50)))
        self.assertEqual(len(config.rl_periodic_checkpoint_paths), 20)
        self.assertEqual(config.rl_periodic_checkpoint_paths[-2].name, "actor_epoch_0500.pt")
        self.assertEqual(config.rl_periodic_checkpoint_paths[-1].name, "critic_epoch_0500.pt")


if __name__ == "__main__":
    unittest.main()
