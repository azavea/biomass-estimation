import torch
from torchvision.transforms import Compose


class SentinelBandNormalize():
    def __init__(self, norm_x=True, norm_y=True):
        self.norm_x = norm_x
        self.norm_y = norm_y

    def __call__(self, item):
        x, y = item

        if self.norm_x:
            x = x.clone()
            s1 = x[:, 0:4, :, :]
            s2 = x[:, 4:14, :, :]
            clp = x[:, 14:15, :, :]

            s1[s1 == -9999] = -50
            s1 += 50
            s1 /= 50

            s2 /= 10_000

            clp[clp == 255] = 100
            clp /= 100

            x = torch.cat([s1, s2, clp], dim=1)

        if y is not None and self.norm_y:
            y = torch.clamp(y, 0, 400)

        return (x, y)


class SelectBands():
    def __init__(self, bands):
        all_bands = list(range(15))
        if bands == 's1':
            bands = all_bands[0:4]
        elif bands == 's2':
            bands = all_bands[4:14]
        elif bands == 'all':
            bands = all_bands
        else:
            raise ValueError(f'{bands} is not valid')
        self.bands = bands

    def __call__(self, item):
        x, y = item
        return (x[:, self.bands, :, :], y)


class AggregateMonths():
    def __init__(self, agg_fn='mean'):
        self.agg_fn = agg_fn
        if self.agg_fn not in ['mean', 'max']:
            raise ValueError(f'{agg_fn} is not valid')

    def __call__(self, item):
        x, y = item
        if self.agg_fn == 'mean':
            x = x.mean(0)
        elif self.agg_fn == 'max':
            x = x.max(0)
        return (x, y)


def build_transform(transform_dicts):
    transforms = []
    for transform_dict in transform_dicts:
        if transform_dict['name'] == 'sentinel_band_normalize':
            transform = SentinelBandNormalize(
                norm_x=transform_dict.get('norm_x', True),
                norm_y=transform_dict.get('norm_y', True))
        elif transform_dict['name'] == 'select_bands':
            transform = SelectBands(transform_dict['bands'])
        elif transform_dict['name'] == 'aggregate_months':
            transform = AggregateMonths(transform_dict['agg_fn'])
        else:
            raise ValueError(f"{transform_dict['name']} is not valid")
        transforms.append(transform)
    return Compose(transforms)
