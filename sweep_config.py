import glob
import wandb
wandb.login()

sweep_config = {
    'method': 'grid',
    'parameters': {
        # our images
        'dataset': {'values': ['div2k_100']},
        'image_id': {'values': [i for i in range(100)]},
        'degradation': {'values': ['gauss_blur', 'white_noise']},
        'severity': {'values': [0, 1, 2, 3, 4, 5, 6, 7]},

        # dataset images
        # 'image_path': {'values': sorted(glob.glob("/home/itaian/group/datasets/LIVE_flattened/*"))},
    }
}
