from types import SimpleNamespace


config_dict = {
    'base_path': "/home/itaian/data/results/SWD/IQA",

    # image pyramid
    'image_id': 0,
    'degradation': 'reference',
    'severity': None,
    'dim_max_size': 1024,
    'num_scales': 9,

    # swd
    'patch_size': 5,
    'num_proj': 64,
    'num_noise': 64,
    'normalized': False,

    # kl
    'eps': 1e-4,

    'hr_noise': True,
    'lr_noise': True,
}

default_config = SimpleNamespace(**config_dict)
