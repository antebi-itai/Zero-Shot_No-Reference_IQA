import torch
from tqdm import tqdm
from images import pt2np, imwrite
import os
import torch
import sys
import wandb
import pickle

from images import imread
from ResizeRight.resize_right import resize

from .degrade import degrade
from .pyramid import pyramid_scale_sqrt_2_shapes
from .swd import best_projection_weights, add_noise, weights_map_to_image
from .swd import get_projection, slow_best_projection_weights, fast_best_projection_weights
from .config import default_config


def our_score(image_paths, dataset_name=None, config=default_config):
    for image_idx in range(len(image_paths)):
        print(f"Image {image_idx} / {len(image_paths)}.")
        image_path = image_paths[image_idx]
        image = imread(image_path).cuda()

        # image  ->  image_pyramid
        image_shapes = pyramid_scale_sqrt_2_shapes(image, num_scales=config.num_scales, patch_size=config.patch_size, dim_max_size=config.dim_max_size)
        normalized_image = resize(image, out_shape=image_shapes[0]).clip(0, 1)
        degraded_normalized_image = degrade(reference=normalized_image, degradation=config.degradation, severity=config.severity)
        image_pyramid = []
        for image_shape in image_shapes:
            image_pyramid.append(resize(degraded_normalized_image, out_shape=image_shape).clip(0, 1))
        print("Image Pyramid Done.")

        # image_pyramid  ->  weight_maps_pyramid
        proj = get_projection(num_proj=config.num_proj, patch_size=config.patch_size, base_path=config.base_path)
        weight_maps_pyramid = []
        edge = config.patch_size // 2
        for hr_idx in [0, (config.num_scales - 2) - 1]:
            hr_image = image_pyramid[hr_idx]
            lr_image = image_pyramid[hr_idx + 2]
            weight_map = slow_best_projection_weights(hr_image=hr_image, lr_image=lr_image, proj=proj, num_noise=config.num_noise, normalized=config.normalized)
            weight_map = weight_map[..., edge:-edge, edge:-edge]
            weight_maps_pyramid.append(weight_map)
        print("Weight Maps Pyramid Done.")

        # weight_maps_pyramid  ->  histogram_pyramid
        histogram_pyramid = []
        for weight_map in weight_maps_pyramid:
            patch_counts = weight_map.flatten().clip(0, 10)
            hist_vals = torch.histc(patch_counts.cpu(), bins=1000, min=0, max=10).cuda()
            hist_vals /= hist_vals.sum()
            histogram_pyramid.append(hist_vals)
        histogram_pyramid = torch.stack(histogram_pyramid)
        print("Histograms Pyramid Done.")

        # histogram_pyramid  ->  score
        kldivs = torch.tensor([kl_divergence(histogram_pyramid[0], histogram_pyramid[-1]),
                               kl_divergence(histogram_pyramid[-1], histogram_pyramid[0])])
        score = kldivs.mean().item()
        return score


def our_iqa_metric(image, config):

    # check if already computed
    result_path = f"{config.base_path}/pickles/image_{config.image_id}_degradation_{config.degradation}_{config.severity}_noises_{config.num_noise}_proj_{config.num_proj}.pkl"
    if os.path.exists(result_path):
        return

    # image  ->  image_pyramid
    image_shapes = pyramid_scale_sqrt_2_shapes(image, num_scales=config.num_scales, patch_size=config.patch_size, dim_max_size=config.dim_max_size)
    normalized_image = resize(image, out_shape=image_shapes[0]).clip(0, 1)
    degraded_normalized_image = degrade(reference=normalized_image, degradation=config.degradation, severity=config.severity)
    image_pyramid = []
    for image_shape in image_shapes:
        image_pyramid.append(resize(degraded_normalized_image, out_shape=image_shape).clip(0, 1))
    print("Image Pyramid Done.")

    # image_pyramid  ->  weight_maps_pyramid
    proj = get_projection(num_proj=config.num_proj, patch_size=config.patch_size, base_path=config.base_path)
    weight_maps_pyramid = []
    edge = config.patch_size // 2
    for hr_idx in [0, (config.num_scales - 2) - 1]:
        hr_image = image_pyramid[hr_idx]
        lr_image = image_pyramid[hr_idx + 2]
        weight_map = slow_best_projection_weights(hr_image=hr_image, lr_image=lr_image, proj=proj, num_noise=config.num_noise, normalized=config.normalized)
        # weight_map = fast_best_projection_weights(hr_image=hr_image, lr_image=lr_image, proj=proj, num_noise=config.num_noise, normalized=config.normalized)
        weight_map = weight_map[..., edge:-edge, edge:-edge]
        weight_maps_pyramid.append(weight_map)
    print("Weight Maps Pyramid Done.")

    # weight_maps_pyramid  ->  histogram_pyramid
    histogram_pyramid = []
    for weight_map in weight_maps_pyramid:
        patch_counts = weight_map.flatten().clip(0, 10)
        hist_vals = torch.histc(patch_counts.cpu(), bins=1000, min=0, max=10).cuda()
        hist_vals /= hist_vals.sum()
        histogram_pyramid.append(hist_vals)
    histogram_pyramid = torch.stack(histogram_pyramid)
    print("Histograms Pyramid Done.")

    # save result
    result = (image_pyramid, weight_maps_pyramid, histogram_pyramid)
    os.makedirs(name=os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(config=default_config):
    with wandb.init(config=config, project="IQA"):
        config = wandb.config
        image = imread(f"{config.base_path}/dataset_huge/{config.image_id:04.0f}.png").cuda()
        our_iqa_metric(image, config)


if __name__ == "__main__":
    print(f"CUDA? environ={os.environ['CUDA_VISIBLE_DEVICES']}, available={torch.cuda.is_available()}")
    if len(sys.argv) > 1:
        sweep_id = sys.argv[1]
        wandb.agent(sweep_id, function=main, project="IQA")
    else:
        main()
