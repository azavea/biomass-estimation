import unittest

import torch

from biomass.models import TemporalUNet


class TestTemporalUNetModels(unittest.TestCase):
    def test1(self):
        model = TemporalUNet(
            15, agg_method='month_attn', time_embed_method='add_inputs',
            encoder_weights=None)
        batch_sz = 2
        times = 12
        channels = 15
        height = 256
        width = 256
        x = torch.randn(batch_sz, times, channels, height, width)
        z = model(x)
        self.assertEqual(z['output'].shape, (batch_sz, 1, height, width))
        self.assertEqual(z['month_weights'].shape, (batch_sz, times))
        self.assertEqual(z['month_outputs'].shape, (batch_sz, times, height, width))


if __name__ == '__main__':
    unittest.main()
