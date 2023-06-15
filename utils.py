import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import portalocker
import os
import pickle
import kornia

from images import pt2np, np2pt


def pt2im(image):
    # pt -> np
    image = pt2np(image)
    # (0, 1) float -> (0, 255) int
    image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    # np -> image
    image = Image.fromarray(image)
    return image


def im2pt(image):
    # image -> np
    image = np.asarray(image, dtype=np.float32)
    # (0, 255) int -> (0, 1) float
    image = np.clip(image / 255.0, 0, 1)
    # np -> pt
    image = np2pt(image)
    return image


def load_pickle(path):
    with open(path, 'rb') as handle:
        result = pickle.load(handle)
    return result


def high_grad_mask(image, patch_size=5, grad_threshold=1/1000):
    dx, dy = kornia.filters.spatial_gradient(image).permute([2, 0, 1, 3, 4])
    grad_map = (dx.pow(2) + dy.pow(2)).sqrt().mean(dim=1, keepdim=True)
    uniform_kernel = torch.ones(1, 1, patch_size, patch_size, device=image.device) / (patch_size*patch_size)
    grad_map_blurred = torch.conv2d(input=grad_map, weight=uniform_kernel)
    mask = grad_map_blurred > grad_threshold
    return mask


# Histograms


def prob_hist(values, bins, min, max):
    values = values.clip(min=min, max=max)
    hist_vals = torch.histc(values.cpu(), bins=bins, min=min, max=max).to(device=values.device)
    hist_vals /= values.numel()
    return hist_vals


def plot_hist(hist, start, end, label=None, color=None):
    if type(hist) is torch.Tensor:
        hist = hist.cpu().numpy()
    assert hist.ndim == 1
    bins = torch.linspace(start=start, end=end, steps=hist.size+1)
    plt.hist(bins[:-1], bins=bins, weights=hist, histtype='step', label=label, color=color)


# KL-divergence


def add_eps_to_hist(hist, percent=1):
    eps_end = (percent / 100) * (1 / hist.numel())
    eps_start = eps_end / (1 - hist.numel()*eps_end)
    hist += eps_start
    hist /= hist.sum()
    return hist


def kl_divergence(p, q):
    r"""The Kullback-Leibler divergence
    Args:
        p: 1-D Tensor of shape (n) of probabilites (sums to 1).
        q: 1-D Tensor of shape (n) of probabilites (sums to 1).
    Return:
        float - KL(p|q)
    """
    assert (type(p) is torch.Tensor) and (type(q) is torch.Tensor)
    assert (p.ndim == 1) and (q.ndim == 1) and (p.numel() == q.numel())
    assert (torch.isclose(p.sum(), torch.tensor(1.))) and (torch.isclose(q.sum(), torch.tensor(1.)))
    # add eps to avoid zeros
    p = add_eps_to_hist(hist=p)
    q = add_eps_to_hist(hist=q)
    # compute kl-div(p|q)
    numel = p.numel()
    kl_div = F.kl_div(q.log(), p, reduction='batchmean') * numel
    return kl_div


def old_kl_divergence(p, q):
    assert (type(p) is torch.Tensor) and (type(q) is torch.Tensor)
    assert (p.ndim == 1) and (q.ndim == 1) and (p.numel() == q.numel())
    assert (torch.isclose(p.sum(), torch.tensor(1.))) and (torch.isclose(q.sum(), torch.tensor(1.)))
    eps = torch.tensor(1e-4)
    p, q = torch.maximum(p, eps), torch.maximum(q, eps)
    # compute kl-div(p|q)
    numel = p.numel()
    kl_div = F.kl_div(q.log(), p, reduction='batchmean') * numel
    return kl_div.item()


def kld_gauss(u1, s1, u2, s2):
  # general KL two Gaussians
  # u2, s2 often N(0,1)
  # https://stats.stackexchange.com/questions/7440/ +
  # kl-divergence-between-two-univariate-gaussians
  # log(s2/s1) + [( s1^2 + (u1-u2)^2 ) / 2*s2^2] - 0.5
  v1 = s1 * s1
  v2 = s2 * s2
  a = np.log(s2/s1)
  num = v1 + (u1 - u2)**2
  den = 2 * v2
  b = num / den
  return a + b - 0.5


# multi-process locking

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def append_to_results(results_path, key, value):
    with portalocker.Lock('.lock', timeout=600) as lock_handle:
        # read
        if os.path.exists(results_path):
            results = read_pickle(file_path=results_path)
        else:
            results = dict()
        # append
        results[key] = value
        # write
        write_pickle(file_path=results_path, data=results)
