from types import SimpleNamespace


config_dict = {
    # sweep
    'image_path': 'error',
    'degradation': 'gauss_blur',
    'severity': 0,
    'results_path': './results/attempt_base',

    # image pyramid
    'hr_dim_size': 1024,
    'min_dim_size': 40,

    # swd
    'patch_size': 5,
    'num_proj': 64,
    'num_noise': 64,
    'normalized': False,

    # kl
    'kl_method': "mean",  # mean, ration

    'debug': False,
}

default_config = SimpleNamespace(**config_dict)
