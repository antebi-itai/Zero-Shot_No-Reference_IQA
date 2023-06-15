import os
import torch
import torch.nn.functional as F
import cupy as cp
from tqdm import tqdm


# projections


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


# multi_bin histograms


def cupy_histogram(x, bins, device='cuda'):
    x = cp.asarray(x)
    bins = cp.asarray(bins)
    hist = cp.histogram(x=x, bins=bins)[0]
    return torch.as_tensor(hist, device=device)


def invert_permutation(permutation):
    output = torch.empty_like(permutation)
    output.scatter_(0, permutation, torch.arange(0, len(permutation), dtype=torch.int64, device=permutation.device))
    return output


def multi_bin_hist(values, bins):
    assert values.ndim == 1 and bins.ndim == 2
    num_bins, num_edges = bins.shape

    flattened_bins = bins.flatten()
    sorted_flattened_bins, sorted_flattened_bins_indices = flattened_bins.sort()

    hist = cupy_histogram(x=values, bins=sorted_flattened_bins)
    cumsum = torch.cat([torch.tensor([0], device=hist.device), torch.cumsum(hist, dim=0)])

    all_indices = invert_permutation(sorted_flattened_bins_indices).reshape(num_bins, num_edges)
    hists = cumsum[all_indices[:, 1:]] - cumsum[all_indices[:, :-1]]
    return hists


# swd weights


def add_noise(tensor, std=0.01, clip=True):
    noise = torch.normal(mean=torch.zeros_like(tensor), std=torch.ones_like(tensor)) * std
    noisy_tensor = tensor + noise
    if clip:
        return noisy_tensor.clip(min=0, max=1)
    return noisy_tensor


def best_projection_weights(hr_image, lr_image, proj, hr_image_mask=None, lr_image_mask=None, normalized=False):
    assert (proj.size(1) == 3) and (proj.size(2) == proj.size(3))
    num_proj = proj.size(0)
    patch_size = proj.size(2)
    assert patch_size % 2 == 1
    mask = True if hr_image_mask is not None else False
    assert not (mask and lr_image_mask is None)

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

    # Mask out patches
    if mask:
        hr_mask = hr_image_mask.flatten()
        lr_mask = lr_image_mask.flatten()
        proj_hr = proj_hr.t()[hr_mask].t()
        proj_lr = proj_lr.t()[lr_mask].t()

    # Find NN indices (hr -> lr)
    proj_lr_sorted, proj_lr_indices = proj_lr.sort(dim=1)
    min_val = min(proj_hr.min(), proj_lr.min()) - 1
    max_val = max(proj_hr.max(), proj_lr.max()) + 1
    lr_bins = torch.cat([min_val * torch.ones(num_proj, 1, device=proj_lr_sorted.device),
                         (proj_lr_sorted[:, 1:] + proj_lr_sorted[:, :-1]) / 2,
                         max_val * torch.ones(num_proj, 1, device=proj_lr_sorted.device)], dim=1)
    weights = torch.stack([torch.as_tensor(cp.histogram(x=cp.asarray(proj_hr[i]), bins=cp.asarray(lr_bins[i]))[0], device='cuda') for i in range(num_proj)])
    if mask:
        assert (weights.sum(dim=1) == proj_hr.size(1)).all()

    # Reshape weights as image for visualization
    _, _, h, w = lr_image.shape
    h_out, w_out = (h - (patch_size - 1)), (w - (patch_size - 1))
    weights_image = torch.stack([weights[i].gather(0, proj_lr_indices[i].argsort(0)) for i in range(num_proj)])
    if mask:
        weights_image_with_mask = torch.nan * torch.empty(num_proj, h_out*w_out, device=weights_image.device)
        weights_image_with_mask.t()[lr_mask] = weights_image.float().t()
        weights_image = weights_image_with_mask
    weights_image = weights_image.reshape(num_proj, 1, h_out, w_out)
    weights_image = F.pad(input=weights_image, pad=[patch_size // 2] * 4, value=0)

    del proj_hr, proj_lr, proj_lr_sorted, proj_lr_indices, lr_bins, weights
    torch.cuda.memory.empty_cache()
    return weights_image


def slow_best_projection_weights(hr_image, lr_image, proj, num_noise=256, hr_image_mask=None, lr_image_mask=None, normalized=False):
    h, w = lr_image.shape[-2:]
    num_proj = proj.size(0)
    weights = torch.zeros(num_proj, 1, h, w)

    # average over noises
    for _ in tqdm(range(num_noise)):
        weights += best_projection_weights(hr_image=add_noise(hr_image), lr_image=add_noise(lr_image), proj=proj,
                                           hr_image_mask=hr_image_mask, lr_image_mask=lr_image_mask, normalized=normalized).float().cpu()
    weights = weights / num_noise

    # average over projections
    weights = weights.mean(dim=0, keepdims=True)

    return weights


def fast_best_projection_weights(hr_image, lr_image, proj, num_noise=256, normalized=False):
    assert (proj.size(1) == 3) and (proj.size(2) == proj.size(3))
    num_proj = proj.size(0)
    patch_size = proj.size(2)
    assert patch_size % 2 == 1

    _, _, h, w = lr_image.shape
    h_out, w_out = (h - (patch_size - 1)), (w - (patch_size - 1))
    final_weights_image = torch.zeros([1, 1, h_out, w_out], device=lr_image.device)

    proj_hr = F.conv2d(hr_image, proj).transpose(0, 1).flatten(start_dim=1)
    noisy_lrs = add_noise(lr_image.repeat([num_noise, 1, 1, 1]))
    noisy_proj_lrs = F.conv2d(noisy_lrs, proj).transpose(0, 1).flatten(start_dim=2)

    for proj_id in tqdm(range(num_proj)):
        noisy_proj_lr = noisy_proj_lrs[proj_id]
        noisy_proj_lr_sorted, noisy_proj_lr_indices = noisy_proj_lr.sort(dim=1)
        min_val = min(proj_hr.min(), noisy_proj_lr.min()) - 1
        max_val = max(proj_hr.max(), noisy_proj_lr.max()) + 1
        lr_bins = torch.cat([min_val * torch.ones(num_noise, 1, device=noisy_proj_lr_sorted.device),
                             (noisy_proj_lr_sorted[:, 1:] + noisy_proj_lr_sorted[:, :-1]) / 2,
                             max_val * torch.ones(num_noise, 1, device=noisy_proj_lr_sorted.device)], dim=1)

        weights = multi_bin_hist(values=proj_hr[proj_id].flatten(), bins=lr_bins)
        # weights = weights.float() / num_noise  # average over hr_noises
        weights = weights.float()
        assert (weights.sum(dim=1) == proj_hr.size(1)).all()

        # Reshape weights as image for visualization
        weights_image = torch.stack([weights[i].gather(0, noisy_proj_lr_indices[i].argsort(0)) for i in range(num_noise)])
        weights_image = weights_image.reshape(num_noise, 1, h_out, w_out)
        weights_image = weights_image.mean(axis=0, keepdim=True)  # average over lr_noises
        final_weights_image += weights_image

    final_weights_image /= num_proj  # average over projections
    final_weights_image = F.pad(input=final_weights_image, pad=[patch_size // 2] * 4, value=0)

    del proj_hr, noisy_lrs, noisy_proj_lr, noisy_proj_lr_sorted, noisy_proj_lr_indices, lr_bins, weights, weights_image
    torch.cuda.memory.empty_cache()
    return final_weights_image


def weights_map_to_image(weights_map, patch_size, pad=False):
    weights_map = (weights_map / 8).clip(0, 1).repeat([1, 3, 1, 1])
    if pad:
        weights_map = F.pad(weights_map, [patch_size // 2] * 4)
    return weights_map
