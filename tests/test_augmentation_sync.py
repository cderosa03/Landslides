import unittest

try:
    import torch
    from dataset.multidata import apply_geometric_transform
except ModuleNotFoundError:
    torch = None
    apply_geometric_transform = None


@unittest.skipIf(torch is None, "PyTorch non installato nell'ambiente")
class AugmentationSyncTest(unittest.TestCase):
    def test_shared_transform_preserves_marker_alignment(self):
        params = {
            "hflip": True,
            "rotation_k": 1,
            "crop": (1, 1, 4, 4),
        }
        marker = torch.zeros((1, 6, 6), dtype=torch.float32)
        marker[:, 2, 3] = 1.0

        planet = apply_geometric_transform(marker, params)
        sentinel = apply_geometric_transform(marker.unsqueeze(0), params)
        mask = apply_geometric_transform(marker, params)

        self.assertEqual(tuple(planet.shape), (1, 4, 4))
        self.assertEqual(tuple(sentinel.shape), (1, 1, 4, 4))
        self.assertTrue(torch.equal(planet, mask))
        self.assertTrue(torch.equal(planet, sentinel.squeeze(0)))


if __name__ == "__main__":
    unittest.main()
