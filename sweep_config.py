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
        'degradation': {'values': ['gauss_blur', 'white_noise']},
        'image_path': {'values': sorted(glob.glob("./images/DIV2K_16/*"))},
        'severity': {'values': [i for i in range(8)]},
    }
}
