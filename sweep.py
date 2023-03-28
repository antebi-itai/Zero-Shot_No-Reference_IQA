import wandb
wandb.login()

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Data
        'image_id': {'values': [i for i in range(30)]},
        'degradation': {'values': ["white_noise", "impulse_noise", "gauss_blur", "glass_blur", "jpeg"]},
        'severity': {'values': [i for i in range(1, 8)]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="IQA")
print("Strated sweep!")
print("Sweep ID:", sweep_id)
