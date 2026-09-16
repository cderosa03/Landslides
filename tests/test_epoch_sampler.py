import unittest

import torch

from dataset.sampler import EpochSubsetSampler, EpochStratifiedSampler


class EpochSubsetSamplerTests(unittest.TestCase):
    def test_exact_count_without_replacement_and_different_epoch_selection(self):
        sampler = EpochSubsetSampler(range(1000), 67, seed=42)
        first = list(sampler)
        self.assertEqual(len(first), 67)
        self.assertEqual(len(set(first)), 67)
        self.assertTrue(all(0 <= index < 1000 for index in first))
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertEqual(len(second), 67)
        self.assertEqual(len(set(second)), 67)
        self.assertNotEqual(set(first), set(second))

    def test_resumed_epoch_reconstructs_selection_without_advancing_previous_epochs(self):
        continuous = EpochSubsetSampler(range(100), 31, seed=7)
        for epoch in range(18):
            continuous.set_epoch(epoch)
            expected = list(continuous)
        resumed = EpochSubsetSampler(range(100), 31, seed=7)
        resumed.set_epoch(17)
        self.assertEqual(list(resumed), expected)

    def test_sampling_does_not_change_global_torch_rng(self):
        before = torch.get_rng_state().clone()
        list(EpochSubsetSampler(range(20), 5))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_full_size_and_invalid_limits(self):
        self.assertEqual(set(EpochSubsetSampler(range(7), 7)), set(range(7)))
        for size in (0, -1, 8):
            with self.assertRaises(ValueError):
                EpochSubsetSampler(range(7), size)

    def test_stratified_sampler_targets_positive_fraction(self):
        class Sample:
            planet_ds = None
            def __len__(self): return 10
            def __getitem__(self, index):
                return {"mask": torch.ones(1) if index < 4 else torch.zeros(1)}
        sampler = EpochStratifiedSampler(Sample(), 8, positive_fraction=0.5)
        selected = list(sampler)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(index < 4 for index in selected), 4)


if __name__ == "__main__":
    unittest.main()
