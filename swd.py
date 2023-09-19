import os
import torch
import torch.nn.functional as F
import cupy as cp
from tqdm import tqdm


# Projections


def sample_random_projections(num_proj, patch_size):
    rand = torch.randn(num_proj, 3 * patch_size ** 2).cuda()
    rand = rand / torch.norm(rand, dim=1, keepdim=True)
    rand = rand.reshape(num_proj, 3, patch_size, patch_size)
    return rand


def get_projection(num_proj, patch_size, base_path="."):
    proj_path = f"{base_path}/projections/{num_proj}_{patch_size}.pt"
    if os.path.exists(proj_path):
        proj = torch.load(proj_path).cuda()
    else:
        proj = sample_random_projections(num_proj=num_proj, patch_size=patch_size)
        os.makedirs(os.path.dirname(proj_path), exist_ok=True)
        torch.save(proj, proj_path)
    return proj


# SWD Weights


def add_noise(tensor, std=0.01, clip=True):
    noise = torch.normal(mean=torch.zeros_like(tensor), std=torch.ones_like(tensor)) * std
    noisy_tensor = tensor + noise
    if clip:
        return noisy_tensor.clip(min=0, max=1)
    return noisy_tensor


def best_projection_weights(hr_image, lr_image, proj, normalized=False):
    assert (proj.size(1) == 3) and (proj.size(2) == proj.size(3))
    num_proj = proj.size(0)
    patch_size = proj.size(2)
    assert patch_size % 2 == 1

    # Project patches
    if not normalized:
        proj_hr = F.conv2d(hr_image, proj).transpose(0, 1).flatten(start_dim=1)
        proj_lr = F.conv2d(lr_image, proj).transpose(0, 1).flatten(start_dim=1)
    else:
        hr_patches = F.unfold(hr_image, kernel_size=patch_size).squeeze()
        lr_patches = F.unfold(lr_image, kernel_size=patch_size).squeeze()
        hr_patches /= hr_patches.norm(dim=0)
        lr_patches /= lr_patches.norm(dim=0)
        proj_hr = torch.mm(proj.flatten(start_dim=1), hr_patches)
        proj_lr = torch.mm(proj.flatten(start_dim=1), lr_patches)

    # Find NN indices (hr -> lr)
    proj_lr_sorted, proj_lr_indices = proj_lr.sort(dim=1)
    min_val = min(proj_hr.min(), proj_lr.min()) - 1
    max_val = max(proj_hr.max(), proj_lr.max()) + 1
    lr_bins = torch.cat([min_val * torch.ones(num_proj, 1, device=proj_lr_sorted.device),
                         (proj_lr_sorted[:, 1:] + proj_lr_sorted[:, :-1]) / 2,
                         max_val * torch.ones(num_proj, 1, device=proj_lr_sorted.device)], dim=1)
    weights = torch.stack([torch.as_tensor(cp.histogram(x=cp.asarray(proj_hr[i]), bins=cp.asarray(lr_bins[i]))[0], device='cuda') for i in range(num_proj)])
    assert (weights.sum(dim=1) == proj_hr.size(1)).all()

    # Reshape weights as image for visualization
    _, _, h, w = lr_image.shape
    h_out, w_out = (h - (patch_size - 1)), (w - (patch_size - 1))
    weights_image = torch.stack([weights[i].gather(0, proj_lr_indices[i].argsort(0)) for i in range(num_proj)])
    weights_image = weights_image.reshape(num_proj, 1, h_out, w_out)
    weights_image = F.pad(input=weights_image, pad=[patch_size // 2] * 4, value=0)

    del proj_hr, proj_lr, proj_lr_sorted, proj_lr_indices, lr_bins, weights
    torch.cuda.memory.empty_cache()
    return weights_image


def stochastic_best_projection_weights(hr_image, lr_image, proj, num_noise=256, normalized=False):
    h, w = lr_image.shape[-2:]
    num_proj = proj.size(0)
    weights = torch.zeros(num_proj, 1, h, w)

    # average over noises
    for _ in tqdm(range(num_noise)):
        weights += best_projection_weights(hr_image=add_noise(hr_image), lr_image=add_noise(lr_image), proj=proj, normalized=normalized).float().cpu()
    weights = weights / num_noise

    # average over projections
    weights = weights.mean(dim=0, keepdims=True)

    return weights


def weights_map_to_image(weights_map, patch_size, pad=False):
    weights_map = (weights_map / 8).clip(0, 1).repeat([1, 3, 1, 1])
    if pad:
        weights_map = F.pad(weights_map, [patch_size // 2] * 4)
    return weights_map
