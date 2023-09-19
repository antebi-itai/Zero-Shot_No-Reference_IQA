from types import SimpleNamespace


config_dict = {
    # sweep
    'results_path': './results/2023_09_19',
    # affects size on disk
    'debug': False,

    # our dataset
    'dataset': 'div2k_100',
    'image_id': 0,
    'degradation': 'gauss_blur',
    'severity': 0,
    'hr_dim_size': 1024,
    # given dataset
    'image_path': '/home/itaian/group/datasets/DIV2K/DIV2K_valid_HR/0803.png',

    # pyramid
    'min_dim_size': 40,
    # swd
    'patch_size': 5,
    'num_proj': 64,
    'num_noise': 64,
    'normalized': False,
    # hist
    'hist_start': 2,
    'hist_end': 8,
    'bins': 1000,
}

default_config = SimpleNamespace(**config_dict)
