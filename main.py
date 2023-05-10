import os
import torch
import wandb
from pathlib import Path

from images import imread
from ResizeRight.resize_right import resize

from IQA.degrade import degrade
from IQA.pyramid import pyramid_shapes_scale_sqrt_2
from IQA.swd import get_projection, slow_best_projection_weights
from IQA.config import default_config
from IQA.utils import write_pickle, kl_divergence, old_kl_divergence


def our_score(image, config):
    # image  ->  image_pyramid
    h, w = image.shape[2:]
    image_shapes = pyramid_shapes_scale_sqrt_2(h=h, w=w, patch_size=config.patch_size, min_dim_size=config.min_dim_size)
    image_pyramid = []
    for image_shape in image_shapes:
        image_pyramid.append(resize(image, out_shape=image_shape).clip(0, 1))
    print("Image Pyramid Done.")

    # image_pyramid  ->  weight_maps_pyramid
    proj = get_projection(num_proj=config.num_proj, patch_size=config.patch_size)
    weight_maps_pyramid = []
    edge = config.patch_size // 2
    num_scales = len(image_pyramid)
    hr_indices = [0, (num_scales - 2) - 1] if not config.debug else [i for i in range(num_scales - 2)]
    for hr_idx in hr_indices:
        hr_image = image_pyramid[hr_idx]
        lr_image = image_pyramid[hr_idx + 2]
        weight_map = slow_best_projection_weights(hr_image=hr_image, lr_image=lr_image, proj=proj, num_noise=config.num_noise, normalized=config.normalized)
        weight_map = weight_map[..., edge:-edge, edge:-edge]
        weight_maps_pyramid.append(weight_map)
    print("Weight Maps Pyramid Done.")

    # weight_maps_pyramid  ->  histogram_pyramid
    histogram_pyramid = []
    for weight_map in weight_maps_pyramid:
        patch_counts = weight_map.flatten().clip(2, 8)
        hist_vals = torch.histc(patch_counts.cpu(), bins=1000, min=2, max=8).cuda()
        hist_vals /= hist_vals.sum()
        histogram_pyramid.append(hist_vals)
    histogram_pyramid = torch.stack(histogram_pyramid)
    print("Histograms Pyramid Done.")

    # histogram_pyramid  ->  score
    kldivs = torch.tensor([old_kl_divergence(histogram_pyramid[0], histogram_pyramid[-1]),
                           old_kl_divergence(histogram_pyramid[-1], histogram_pyramid[0])])
    if config.kl_method == "mean":
        score = kldivs.mean().item()
    elif config.kl_method == "ration":
        score = (kldivs[0] / kldivs[1]).item()
    else:
        raise NotImplementedError
    print("KL-div Done.")

    results_dict = {'image_pyramid': image_pyramid,
                    'weight_maps_pyramid': weight_maps_pyramid,
                    'histogram_pyramid': histogram_pyramid,
                    'kldivs': kldivs,
                    'score': score}

    return results_dict


def main(config=default_config):
    with wandb.init(config=config, project="IQA"):
        config = wandb.config

        # get input degraded image
        original_image = imread(config.image_path).cuda()
        h, w = original_image.shape[2:]
        downscale = config.hr_dim_size / max(h, w)
        assert downscale <= 1, "image too small"
        hr_image = resize(original_image, scale_factors=downscale)
        degraded_image = degrade(reference=hr_image, degradation=config.degradation, severity=config.severity)
        degraded_image = (degraded_image * 255).round() / 255

        # run our algorithm
        results_dict = our_score(image=degraded_image, config=config)

        # save result
        input_image_name = os.path.basename(config.image_path).split(".")[0]
        output_image_name = f"{config.degradation}--{input_image_name}--{config.severity}.pkl"
        output_image_path = os.path.join(config.results_path, "pkl", output_image_name)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        write_pickle(file_path=output_image_path, data=results_dict)


if __name__ == "__main__":
    main()
