import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


try:
    make_patches = importlib.import_module("s2_builder.make_patches")
except ModuleNotFoundError:
    make_patches = None


@unittest.skipIf(make_patches is None, "geospatial dependencies are unavailable")
class SentinelMosaicTests(unittest.TestCase):
    def test_mosaic_uses_first_valid_tile_in_sorted_order(self):
        first_coverage = np.array([[True, False], [True, False]])
        second_coverage = np.array([[False, True], [True, True]])
        first_10m = np.ones((4, 2, 2), dtype=np.float32)
        first_20m = np.ones((6, 2, 2), dtype=np.float32)
        second_10m = np.full((4, 2, 2), 2.0, dtype=np.float32)
        second_20m = np.full((6, 2, 2), 2.0, dtype=np.float32)

        with patch.object(
            make_patches,
            "reproject_s2_to_patch",
            side_effect=[
                (first_10m, first_coverage),
                (first_20m, first_coverage),
                (second_10m, second_coverage),
                (second_20m, second_coverage),
            ],
        ):
            mosaic_10m, mosaic_20m, valid = make_patches.mosaic_s2_images(
                [{"10m": Path("first_10m.tif"), "20m": Path("first_20m.tif")},
                 {"10m": Path("second_10m.tif"), "20m": Path("second_20m.tif")}],
                object(),
            )

        expected = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(mosaic_10m[0], expected)
        np.testing.assert_array_equal(mosaic_20m[0], expected)
        np.testing.assert_array_equal(valid, np.ones((2, 2), dtype=bool))
