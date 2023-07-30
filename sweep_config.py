import glob
import wandb
wandb.login()

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        # our images
        'dataset': {'values': ['div2k_100']},
        'image_id': {'values': [i for i in range(100)]},
        'degradation': {'values': ['gauss_blur', 'white_noise']},
        'severity': {'values': [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7]},
        'post_degradation': {'values': ['original', 'blur_0.4', 'noise_0.4']},
        # dataset images
        # 'image_path': {'values': sorted(glob.glob("/home/itaian/group/datasets/LIVE_flattened/*"))},
    }
}
