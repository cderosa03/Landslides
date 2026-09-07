import importlib
import unittest
from unittest.mock import patch


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    swinunet = importlib.import_module("models.swinunet")
except ModuleNotFoundError:
    swinunet = None


EncoderBase = nn.Module if swinunet is not None else object

@unittest.skipIf(swinunet is None, "PyTorch/timm are unavailable")
class AuxIntegrationTests(unittest.TestCase):
    class FakeSwinEncoder(EncoderBase):
        def __init__(self, model_name, img_size, in_chans, pretrained, out_indices):
            super().__init__()
            self.out_channels = [3, 6, 12, 24]
            self.layers = nn.ModuleList(
                nn.Conv2d(input_channels, output_channels, kernel_size=1)
                for input_channels, output_channels in zip(
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

    def test_aux_changes_output_and_receives_gradients(self):
        with patch.object(swinunet, "SwinEncoder", self.FakeSwinEncoder):
            model = swinunet.ChangeDetectionSwinUNet(img_size=32, model_size="tiny")
        model.eval()
        s2 = torch.randn(1, 1, 10, 32, 32)
        planet = torch.randn(1, 3, 32, 32)
        aux = torch.zeros(1, 4, 32, 32)

        output_without_aux = model(s2, s2 + 0.1, planet, planet + 0.1, aux)
        output_with_aux = model(s2, s2 + 0.1, planet, planet + 0.1, aux + 0.5)
        self.assertFalse(torch.allclose(output_without_aux, output_with_aux))

        output_with_aux.mean().backward()
        gradient = model.aux_encoder.stem[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.count_nonzero(gradient).item() > 0)
        for encoder in (model.s2_encoder, model.planet_encoder):
            self.assertTrue(
                any(
                    parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
                    for parameter in encoder.parameters()
                )
            )
