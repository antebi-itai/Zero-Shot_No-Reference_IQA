import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from images import pt2np, np2pt


# Images


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


def kl_divergence(p, q):
    assert (type(p) is torch.Tensor) and (type(q) is torch.Tensor)
    assert (p.ndim == 1) and (q.ndim == 1) and (p.numel() == q.numel())
    assert (torch.isclose(p.sum(), torch.tensor(1.))) and (torch.isclose(q.sum(), torch.tensor(1.)))
    eps = torch.tensor(1e-4)
    p, q = torch.maximum(p, eps), torch.maximum(q, eps)
    # compute kl-div(p|q)
    numel = p.numel()
    kl_div = F.kl_div(q.log(), p, reduction='batchmean') * numel
    return kl_div.item()


# Pickles

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
