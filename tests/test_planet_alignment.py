import contextlib
import unittest

try:
    import numpy as np
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from build_ps_dataset import align_to_reference, alignment_issues
except ImportError:
    np = None


@unittest.skipIf(np is None, "Dipendenze geospaziali non installate")
class PlanetAlignmentTest(unittest.TestCase):
    def test_shifted_raster_is_reprojected_to_reference_grid(self):
        profile = {
            "driver": "GTiff",
            "height": 4,
            "width": 4,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:32632",
        }
        with MemoryFile() as ref_file, MemoryFile() as shifted_file:
            with ref_file.open(
                **profile, transform=from_origin(500000, 5100000, 10, 10)
            ) as reference, shifted_file.open(
                **profile, transform=from_origin(500010, 5100000, 10, 10)
            ) as shifted:
                reference.write(np.ones((1, 4, 4), dtype=np.uint8))
                shifted.write(np.ones((1, 4, 4), dtype=np.uint8))

                self.assertIn("transform affine", alignment_issues(reference, shifted))
                with contextlib.ExitStack() as stack:
                    aligned = align_to_reference(
                        stack, shifted, reference, "post", Resampling.bilinear
                    )
                    self.assertEqual(aligned.crs, reference.crs)
                    self.assertEqual(aligned.transform, reference.transform)
                    self.assertEqual((aligned.height, aligned.width), (4, 4))


if __name__ == "__main__":
    unittest.main()
