import os
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader, Dataset

from dataset.lands2 import PSLandslideSentinel2Dataset
from dataset.landslides import PSLandslideDataset, REQUIRED_FILENAMES
from train import seed_worker
import train as training
from utils.train_runtime import raster_environment


class RasterFixture(Dataset):
    def __init__(self, path):
        self.path = str(path)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        from rasterio.env import get_gdal_config
        with rasterio.open(self.path) as source:
            values = torch.from_numpy(source.read(out_dtype="float32"))
        return values, get_gdal_config("GDAL_CACHEMAX"), torch.cuda.is_initialized()


class LoaderStabilityTests(unittest.TestCase):
    @staticmethod
    def make_index_fixture(root):
        for index in range(1, 11):
            directory = root / "EmiliaRomagna2023" / str(index)
            directory.mkdir(parents=True)
            for filename in REQUIRED_FILENAMES:
                (directory / filename).touch()
            for date in ("20230501", "20230520"):
                temporal = directory / "s2" / date
                temporal.mkdir(parents=True)
                for filename in ("s2_10m.tif", "s2_20m.tif", "s2_valid.tif"):
                    (temporal / filename).touch()
        (root / "EmiliaRomagna2023/1/s2/20230520/s2_valid.tif").unlink()
        (root / "EmiliaRomagna2023/2/mask.tif").unlink()

    def test_bounded_index_skips_incomplete_pairs_and_stops_before_rest_of_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_index_fixture(root)
            original_iterdir = Path.iterdir

            def bounded_iterdir(path):
                if path.name == "s2" and int(path.parent.name) >= 5:
                    raise AssertionError("Smoke probe scanned beyond its valid sample limit")
                return original_iterdir(path)

            with patch("dataset.lands2.random.Random.shuffle", lambda rng, items: None), \
                 patch.object(Path, "iterdir", bounded_iterdir):
                dataset = PSLandslideSentinel2Dataset(
                    root, ["EmiliaRomagna2023"], sample_limit=2, required_files=REQUIRED_FILENAMES,
                )
            self.assertEqual([sample["patch_id"] for sample in dataset.samples], ["3", "4"])
            self.assertEqual(dataset.excluded["incomplete_pre_post_temporal_pair"], 1)
            self.assertEqual(dataset.excluded["missing_required_planet_file"], 1)
            full = PSLandslideSentinel2Dataset(root, ["EmiliaRomagna2023"])
            self.assertEqual(len(full.samples), 9)

    def test_smoke_selection_is_reproducible_and_planet_only_visits_selected_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_index_fixture(root / "patches")
            args = training.parse_args(["--description", "index fixture", "--dataset-root", str(root / "patches")])
            previous = Path.cwd()
            original_iterdir = Path.iterdir

            def selected_iterdir(path):
                if path.name == "EmiliaRomagna2023":
                    raise AssertionError("Planet must not enumerate the full event for a smoke probe")
                return original_iterdir(path)

            try:
                os.chdir(root)
                with patch.object(Path, "iterdir", selected_iterdir):
                    planet, sentinel = training.build_event_datasets(args, ["EmiliaRomagna2023"], sample_limit=3)
                    _, repeated = training.build_event_datasets(args, ["EmiliaRomagna2023"], sample_limit=3)
                self.assertEqual(len(planet), 3)
                keys = [sample["patch_id"] for sample in sentinel.samples]
                self.assertEqual(keys, [sample["patch_id"] for sample in repeated.samples])
                self.assertEqual(set(keys), {sample["patch_id"] for sample in planet.folders})
                self.assertTrue(set(keys).isdisjoint({"1", "2"}))
            finally:
                os.chdir(previous)

    def test_smoke_runs_both_phases_and_keeps_results_out_of_experiment_table(self):
        class TinyModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.layer = torch.nn.Conv2d(3, 1, 1)

            def forward(self, s2_pre, s2_post, planet_pre, planet_post, aux, **kwargs):
                return self.layer(planet_pre)

        class TinyDataset(Dataset):
            def __len__(self):
                return 12

            def __getitem__(self, index):
                mask = torch.zeros(1, 8, 8, dtype=torch.uint8)
                mask[:, 2:5, 2:5] = 1
                return {
                    "planet_pre": torch.ones(3, 8, 8), "planet_post": torch.ones(3, 8, 8),
                    "aux": torch.zeros(4, 8, 8), "mask": mask,
                    "s2_pre": torch.ones(1, 10, 8, 8), "s2_post": torch.ones(1, 10, 8, 8),
                    "s2_valid_pre": torch.ones(1, dtype=torch.bool),
                    "s2_valid_post": torch.ones(1, dtype=torch.bool),
                    "event": "SyntheticEvent", "patch_id": str(index),
                }

        import json
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                with patch.object(training, "ChangeDetectionSwinUNet", TinyModel), \
                     patch.object(training, "PSLandslideDataset"), \
                     patch.object(training, "PSLandslideSentinel2Dataset"), \
                     patch.object(training, "MultiModalLandslideDataset", return_value=TinyDataset()), \
                     patch.object(training.torch.cuda, "is_available", return_value=False):
                    self.assertEqual(training.main([
                        "--description", "diagnostic fixture", "--num-workers", "0",
                        "--smoke-batches", "2", "--batch-size", "2", "--val-batch-size", "2",
                    ]), 0)
                experiment = next(Path("exp").glob("*_smoke"))
                config = json.loads((experiment / "config.json").read_text())
                self.assertTrue(config["diagnostic_only"])
                self.assertEqual(config["epochs"], 1)
                self.assertEqual(config["loader_timeout"], 0)
                self.assertTrue((experiment / "checkpoint_last.pth").exists())
                self.assertTrue((experiment / "history.csv").exists())
                results = json.loads((experiment / "results.json").read_text())
                self.assertGreater(results["best_F1"], 0)
                self.assertLessEqual(results["best_F1"], 1)
                self.assertFalse(Path("exp/experiments.csv").exists())
                log = next(Path("run_logs").glob("*/training.log")).read_text()
                self.assertIn("SMOKE TEST PASSED", log)
                self.assertIn("Validation batch", log)
            finally:
                os.chdir(previous)

    def test_spawn_reads_rasters_with_bounded_cache_after_parent_gdal_init(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tif"
            expected = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
            with rasterio.open(path, "w", driver="GTiff", height=8, width=8,
                               count=1, dtype="float32", transform=from_origin(1, 2, 1, 1)) as target:
                target.write(expected)
            with raster_environment(32):
                loader = DataLoader(RasterFixture(path), batch_size=2, num_workers=2,
                                    multiprocessing_context="spawn", prefetch_factor=1,
                                    timeout=60, worker_init_fn=partial(seed_worker, gdal_cache_mb=16))
                batches = list(loader)
            self.assertEqual(len(batches), 2)
            for values, cache_bytes, cuda_initialized in batches:
                np.testing.assert_array_equal(values[0].numpy(), expected)
                self.assertTrue(torch.all(cache_bytes == 16 * 1024**2))
                self.assertFalse(torch.any(cuda_initialized))

    def test_sentinel_preallocation_preserves_values_and_zero_padding(self):
        dataset = PSLandslideSentinel2Dataset.__new__(PSLandslideSentinel2Dataset)
        dataset.n_temporal = 6
        frames = [torch.full((10, 8, 8), 1000.0), torch.full((10, 8, 8), 2000.0)]
        with patch.object(dataset, "_load_frame", side_effect=frames):
            stack, valid = dataset._build_stack(Path("unused"), ["20200101", "20200102"])
        expected = torch.stack(frames + [torch.zeros_like(frames[0]) for _ in range(4)])
        torch.testing.assert_close(stack, expected)
        self.assertEqual(valid.tolist(), [True, True, False, False, False, False])
        stack.div_(10000)
        torch.testing.assert_close(stack, expected / 10000)
        self.assertEqual(frames[0][0, 0, 0].item(), 1000)

    def test_sentinel_nodata_conversion_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nodata.tif"
            values = np.array([[[1000, -9999], [0, 2000]]], dtype=np.int16)
            with rasterio.open(path, "w", driver="GTiff", height=2, width=2,
                               count=1, dtype="int16", nodata=-9999,
                               transform=from_origin(1, 2, 1, 1)) as target:
                target.write(values)
            dataset = PSLandslideSentinel2Dataset.__new__(PSLandslideSentinel2Dataset)
            result = dataset._read(path)
            torch.testing.assert_close(result, torch.tensor([[[1000., 0.], [0., 2000.]]]))

    def test_planet_index_is_reused_after_json_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "patches" / "Event" / "1"
            directory.mkdir(parents=True)
            for filename in ("pre.tif", "post.tif", "dem.tif", "slope.tif", "aspect.tif", "mask.tif"):
                (directory / filename).touch()
            previous = Path.cwd()
            try:
                os.chdir(root)
                initial = PSLandslideDataset(root / "patches", ["Event"], 128)
                with patch.object(PSLandslideDataset, "_read_folders", side_effect=AssertionError("cache discarded")):
                    reused = PSLandslideDataset(root / "patches", ["Event"], 128)
                self.assertEqual(initial.folders, reused.folders)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
