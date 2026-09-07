import importlib
import unittest


try:
    import torch

    swinunet = importlib.import_module("models.swinunet")
except ModuleNotFoundError:
    swinunet = None


@unittest.skipIf(swinunet is None, "PyTorch/timm are unavailable")
class DecoderShapeTests(unittest.TestCase):
    def test_decoder_uses_spatial_upsampling_for_supported_sizes(self):
        for image_size in (64, 128):
            decoder = swinunet.TransformerDecoder(
                img_size=image_size,
                enc_channels=(3, 6, 12, 24),
                depths_decoder=(0, 0, 0, 0),
                num_heads=(3, 6, 12, 24),
            )
            features = [
                torch.randn(1, channels, image_size // scale, image_size // scale)
                for channels, scale in zip((3, 6, 12, 24), (4, 8, 16, 32))
            ]
            self.assertEqual(tuple(decoder(features).shape), (1, 1, image_size, image_size))

    def test_localized_marker_keeps_its_spatial_position(self):
        image_size = 32
        decoder = swinunet.TransformerDecoder(
            img_size=image_size,
            enc_channels=(2, 2, 2, 2),
            depths_decoder=(0, 0, 0, 0),
            num_heads=(1, 1, 1, 1),
        )
        with torch.no_grad():
            # At every fusion, retain the skip feature and discard the
            # upsampled deeper feature.
            for projection in decoder.concat_back_dim[:-1]:
                projection.weight.zero_()
                projection.bias.zero_()
                projection.weight[:, 2:] = torch.eye(2)
            decoder.output.weight.zero_()
            decoder.output.bias.zero_()
            decoder.output.weight[0, 0, 0, 0] = 1.0

        features = [
            torch.zeros(1, 2, image_size // scale, image_size // scale)
            for scale in (4, 8, 16, 32)
        ]
        marker_row, marker_col = 3, 5
        features[0][0, 0, marker_row, marker_col] = 1.0
        output = decoder(features)[0, 0]

        maxima = torch.nonzero(output == output.max(), as_tuple=False).float().mean(0)
        expected = torch.tensor(
            [(marker_row + 0.5) * 4 - 0.5, (marker_col + 0.5) * 4 - 0.5]
        )
        self.assertTrue(torch.all(torch.abs(maxima - expected) <= 2.0))
