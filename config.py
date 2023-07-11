from types import SimpleNamespace


config_dict = {
    # sweep
    'results_path': './results/NewImagePyramid',

    'image_path': './images/DIV2K_16/0001.png',
    'degradation': 'gauss_blur',
    'severity': 0,

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

    'debug': True,
}

default_config = SimpleNamespace(**config_dict)
