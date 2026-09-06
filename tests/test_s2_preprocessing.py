import ast
import importlib
import tempfile
import unittest
from pathlib import Path


try:
    combine_bands = importlib.import_module("s2_builder.combine_bands")
except ModuleNotFoundError:
    combine_bands = None


ROOT = Path(__file__).resolve().parents[1]


class SentinelPreprocessingStaticTests(unittest.TestCase):
    def test_ten_l2a_bands_and_lazy_credentials_are_declared(self):
        combine_tree = ast.parse(
            (ROOT / "s2_builder" / "combine_bands.py").read_text(encoding="utf-8")
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in combine_tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"PRODUCTS_10m", "PRODUCTS_20m"}
        }
        self.assertEqual(
            assignments["PRODUCTS_10m"] + assignments["PRODUCTS_20m"],
            [
                "B02_10m", "B03_10m", "B04_10m", "B08_10m",
                "B05_20m", "B06_20m", "B07_20m", "B8A_20m", "B11_20m", "B12_20m",
            ],
        )

        download_tree = ast.parse(
            (ROOT / "s2_builder" / "S2_download.py").read_text(encoding="utf-8")
        )
        module_assignments = {
            target.id
            for node in download_tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("credentials", module_assignments)

    def test_download_windows_span_every_event_cutoff(self):
        tree = ast.parse(
            (ROOT / "s2_builder" / "S2_download.py").read_text(encoding="utf-8")
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"EVENT_CUTOFFS", "INVENTORIES"}
        }
        for inventory in assignments["INVENTORIES"]:
            cutoff = assignments["EVENT_CUTOFFS"][inventory["name"]]
            self.assertLess(inventory["start_date"][:10], cutoff)
            self.assertLess(cutoff, inventory["end_date"][:10])


@unittest.skipIf(combine_bands is None, "geospatial dependencies are unavailable")
class SentinelL2APreprocessingTests(unittest.TestCase):
    def test_l2a_band_lookup_requires_exact_band_and_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            product_dir = Path(temporary_directory)
            expected = product_dir / "T32TQQ_20230501T101031_B02_10m.jp2"
            expected.touch()
            (product_dir / "T32TQQ_20230501T101031_B02_20m.jp2").touch()
            (product_dir / "T32TQQ_20230501T101031_B12_20m.jp2").touch()

            self.assertEqual(
                combine_bands.find_l2a_band(product_dir, "B02", "10m"), expected
            )
            with self.assertRaises(RuntimeError):
                combine_bands.find_l2a_band(product_dir, "B03", "10m")
