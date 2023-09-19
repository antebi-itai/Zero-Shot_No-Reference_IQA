import os
import torch
import wandb
from images import imread
from ResizeRight.resize_right import resize
from IQA.config import default_config
from IQA.degrade import degrade
from IQA.pyramid import create_image_pyramids
from IQA.swd import get_projection, stochastic_best_projection_weights
from IQA.utils import write_pickle, kl_divergence


def our_score(image, config):
    # image  ->  image_pyramid
    image_pyramid, image_pyramid_with_borders, full_size_image_pyramid = create_image_pyramids(image=image, patch_size=config.patch_size, min_dim_size=config.min_dim_size)
    print("Image Pyramid Done.")

    # image_pyramid  ->  weight_maps_pyramid
    proj = get_projection(num_proj=config.num_proj, patch_size=config.patch_size)
    weight_maps_pyramid = []
    edge = config.patch_size // 2
    num_scales = len(image_pyramid)
    hr_indices = [0, (num_scales - 1) - 2] if not config.debug else [i for i in range(num_scales - 2)]
    for hr_idx in hr_indices:
        hr_image = image_pyramid_with_borders[hr_idx]
        lr_image = image_pyramid_with_borders[hr_idx + 2]
        weight_map = stochastic_best_projection_weights(hr_image=hr_image, lr_image=lr_image, proj=proj, num_noise=config.num_noise, normalized=config.normalized)
        weight_map = weight_map[..., edge:-edge, edge:-edge]
        weight_maps_pyramid.append(weight_map)
    print("Weight Maps Pyramid Done.")

    # weight_maps_pyramid  ->  histogram_pyramid
    histogram_pyramid = []
    for weight_map in weight_maps_pyramid:
        patch_counts = weight_map.flatten().clip(config.hist_start, config.hist_end)
        hist_vals = torch.histc(patch_counts.cpu(), bins=config.bins, min=config.hist_start, max=config.hist_end).cuda()
        hist_vals /= hist_vals.sum()
        histogram_pyramid.append(hist_vals)
    histogram_pyramid = torch.stack(histogram_pyramid)
    print("Histograms Pyramid Done.")

    # histogram_pyramid  ->  score
    kldivs = torch.tensor([kl_divergence(histogram_pyramid[0], histogram_pyramid[-1]),
                           kl_divergence(histogram_pyramid[-1], histogram_pyramid[0])])
    print("KL-div Done.")

    # result
    kl1, kl2 = kldivs
    results_dict = {'kl1': kl1,
                    'kl2': kl2}
    if config.debug:
        results_dict.update(
            {'image_pyramid': image_pyramid,
             'image_pyramid_with_borders': image_pyramid_with_borders,
             'full_size_image_pyramid': full_size_image_pyramid,
             'weight_maps_pyramid': weight_maps_pyramid,
             'histogram_pyramid': histogram_pyramid})

    return results_dict


def main_our_images(config=default_config):
    with wandb.init(config=config, project="IQA"):
        config = wandb.config

        # get input degraded image
        image_path = f'./images/{config.dataset}/{config.image_id:04d}.png'
        original_image = imread(image_path).cuda()
        h, w = original_image.shape[2:]
        downscale = config.hr_dim_size / max(h, w)
        if downscale < 1:
            hr_image = resize(original_image, scale_factors=downscale, pad_mode='reflect')
        else:
            hr_image = original_image
        degraded_image = degrade(reference=hr_image, degradation=config.degradation, severity=config.severity)

        # run our algorithm
        results_dict = our_score(image=degraded_image, config=config)

        # save result
        output_image_name = f"DATASET_{config.dataset}_IMAGE_{config.image_id:04d}_DEGRADATION_{config.degradation}_SEVERITY_{config.severity}.pkl"
        output_image_path = os.path.join(config.results_path, "pkl", output_image_name)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        write_pickle(file_path=output_image_path, data=results_dict)


def main_dataset_images(config=default_config):
    with wandb.init(config=config, project="IQA"):
        config = wandb.config

        original_image = imread(config.image_path).cuda()
        results_dict = our_score(image=original_image, config=config)
        input_image_name = os.path.basename(config.image_path)[:-4]
        output_image_path = os.path.join(config.results_path, "pkl", f"{input_image_name}.pkl")
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        write_pickle(file_path=output_image_path, data=results_dict)


if __name__ == "__main__":
    main_our_images()
