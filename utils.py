import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import portalocker
import os
import pickle
import glob
from scipy.stats import spearmanr, pearsonr

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


def warp_hist(hist, indices):
    hist_cumsum = torch.cumsum(hist, dim=0)
    hist_cumsum_slopes = torch.cat([hist[1:], torch.tensor([0])])
    int_indices = indices.floor().long()
    dec_indices = indices - int_indices
    hist_cumsum_sampled = hist_cumsum[int_indices] + dec_indices * hist_cumsum_slopes[int_indices]
    hist_sampled = torch.cat([hist_cumsum_sampled[:1], hist_cumsum_sampled[1:] - hist_cumsum_sampled[:-1]])
    return hist_sampled


def get_indices(hist_a, hist_b):
    histograms = torch.stack([hist_a, hist_b], dim=0)
    histograms_cumsum = torch.cumsum(histograms, dim=-1)
    indices = torch.searchsorted(histograms_cumsum[0], histograms_cumsum[1])

    # fix residual indices
    residual = histograms_cumsum[0][indices] - histograms_cumsum[1]
    hist_cumsum_slopes = torch.cat([hist_a[indices][1:], torch.tensor([1])])
    residual_indices = torch.where(hist_cumsum_slopes == 0, torch.tensor(0, dtype=torch.float32),
                                   residual / hist_cumsum_slopes)
    final_indices = indices.float() - residual_indices

    return final_indices


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


def plot_results(images_dir="./images/DIV2K_16/*", result_dir="./results/TrimEdgesPyramid/",
                 degradations=['gauss_blur', 'white_noise'], severities=[i for i in range(8)],
                 score_name='score_mean'):
    image_names = [os.path.basename(image_path).split(".")[0] for image_path in sorted(glob.glob(images_dir))]

    for degradation in degradations:
        # read results data
        scores = torch.zeros(len(image_names), len(severities))
        for image_id, image_name in enumerate(image_names):
            for severity in severities:
                scores[image_id, severity] = \
                    read_pickle(f"{result_dir}/pkl/{degradation}--{image_id + 1:04d}--{severity}.pkl")[score_name]

        # mean-images graph
        score_mean = scores.mean(dim=0)
        score_var = scores.var(dim=0)
        plt.figure()
        plt.errorbar(severities, score_mean, score_var)
        plt.xlabel('severity')
        plt.ylabel('score')
        plt.title(f"{degradation}_all_images")
        out_path = f"{result_dir}/graphs/{degradation}_{score_name}_all_images.png"
        os.makedirs(name=os.path.dirname(out_path), exist_ok=True)
        plt.savefig(fname=out_path, facecolor="white")
        plt.clf()


def plot_dataset_results(result_dir, degradations=['gauss_blur', 'white_noise']):
    # scores
    gt_scores, pred_scores = [], []
    for result_name in sorted(os.listdir(f"{result_dir}/pkl/")):
        gt_score, image_name, image_degradation, _ = result_name.split('__')
        if image_degradation in degradations:
            pred_score = read_pickle(f"{result_dir}/pkl/{result_name}")['score'].cpu()
            gt_scores.append(gt_score)
            pred_scores.append(pred_score)
    # metrics
    srocc = spearmanr(gt_scores, pred_scores).correlation
    plcc = pearsonr(gt_scores, pred_scores)[0]
    # plot
    plt.figure()
    plt.scatter(gt_scores, pred_scores, s=1)
    plt.title(f"{'_and_'.join(degradations)}\nSROCC/PLCC = {round(srocc, 3)}/{round(plcc, 3)}")
    plt.xlabel("gt score")
    plt.ylabel(f"pred score")
    # save plot
    result_path = f"{result_dir}/dataset_graphs/{'_and_'.join(degradations)}.png"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    plt.savefig(result_path)
    plt.close()
