import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    import numpy as np
    import rasterio
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from rasterio.transform import from_origin
    from torch.utils.data import DataLoader, Dataset

    inference = importlib.import_module("inference")
    swinunet = importlib.import_module("models.swinunet")
    training = importlib.import_module("train")
except ModuleNotFoundError:
    torch = None


BaseModule = nn.Module if torch is not None else object
BaseDataset = Dataset if torch is not None else object


@unittest.skipIf(torch is None, "PyTorch/timm/rasterio are unavailable")
class SyntheticEndToEndTests(unittest.TestCase):
    class FakeSwinEncoder(BaseModule):
        def __init__(self, model_name, img_size, in_chans, pretrained, out_indices):
            super().__init__()
            self.out_channels = [3, 6, 12, 24]
            self.layers = nn.ModuleList(
                nn.Conv2d(source, target, kernel_size=1)
                for source, target in zip(
                    [in_chans, 3, 6, 12], self.out_channels
                )
            )

        def forward(self, inputs):
            current = functional.avg_pool2d(inputs, kernel_size=4)
            features = []
            for index, layer in enumerate(self.layers):
                current = layer(current)
                features.append(current)
                if index < len(self.layers) - 1:
                    current = functional.avg_pool2d(current, kernel_size=2)
            return features

    class Fixture(BaseDataset):
        def __init__(self):
            generator = torch.Generator().manual_seed(7)
            self.samples = []
            for index in range(2):
                mask = torch.zeros(1, 32, 32, dtype=torch.uint8)
                mask[:, 8 + index:16 + index, 10:18] = 1
                self.samples.append({
                    "planet_pre": torch.rand(3, 32, 32, generator=generator),
                    "planet_post": torch.rand(3, 32, 32, generator=generator),
                    "aux": torch.rand(4, 32, 32, generator=generator),
                    "s2_pre": torch.rand(1, 10, 32, 32, generator=generator),
                    "s2_post": torch.rand(1, 10, 32, 32, generator=generator),
                    "s2_valid_pre": torch.ones(1, dtype=torch.bool),
                    "s2_valid_post": torch.ones(1, dtype=torch.bool),
                    "mask": mask,
                    "event": "SyntheticEvent",
                    "patch_id": str(index + 1),
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    class NullWriter:
        def add_scalar(self, *args, **kwargs):
            pass

    def test_training_validation_checkpoint_and_geotiff_inference(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            swinunet, "SwinEncoder", self.FakeSwinEncoder
        ):
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            training.EXPERIMENT_DIR = experiment
            training.RUN_CONFIG = {
                "description": "synthetic smoke test",
                "model": "swinunet",
                "model_size": "tiny",
                "patch_size": 32,
            }
            training.writer = self.NullWriter()
            training.device = torch.device("cpu")
            training.START_EARLY_STOPPING_FROM_EPOCH = 2

            model = swinunet.ChangeDetectionSwinUNet(
                img_size=32, model_size="tiny", pretrained=False
            )
            loader = DataLoader(self.Fixture(), batch_size=2, shuffle=False)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=1
            )
            training.train(
                model, loader, loader, criterion, optimizer, scheduler, epochs=1
            )

            checkpoint_path = experiment / "best_model.pth"
            self.assertTrue(checkpoint_path.exists())
            restored, checkpoint = inference.load_model(checkpoint_path)
            probability = inference.predict(restored, loader.dataset[0])
            self.assertEqual(probability.shape, (32, 32))
            self.assertTrue(np.isfinite(probability).all())

            reference = root / "pre.tif"
            transform = from_origin(500000, 5100000, 3, 3)
            with rasterio.open(
                reference,
                "w",
                driver="GTiff",
                height=32,
                width=32,
                count=3,
                dtype="uint8",
                crs="EPSG:32632",
                transform=transform,
            ) as destination:
                destination.write(np.zeros((3, 32, 32), dtype=np.uint8))

            outputs = root / "outputs"
            inference.write_outputs(
                reference,
                outputs,
                probability,
                checkpoint["best_threshold"],
                checkpoint_path,
            )
            for filename, dtype in (("probability.tif", "float32"), ("mask.tif", "uint8")):
                with rasterio.open(outputs / filename) as result:
                    self.assertEqual(result.crs.to_string(), "EPSG:32632")
                    self.assertEqual(result.transform, transform)
                    self.assertEqual(result.dtypes[0], dtype)


if __name__ == "__main__":
    unittest.main()
