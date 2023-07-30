from types import SimpleNamespace


config_dict = {
    # sweep
    'results_path': './results/2023_07_16',

    'dataset': 'div2k_100',
    'image_id': 0,
    'degradation': 'gauss_blur',
    'severity': 0,
    'post_degradation': 'original',

    # image pyramid
    'hr_dim_size': 1024,  # 1024, 500
    'min_dim_size': 40,

    # swd
    'patch_size': 5,
    'num_proj': 64,
    'num_noise': 64,
    'normalized': False,

    'debug': False,
}

default_config = SimpleNamespace(**config_dict)
