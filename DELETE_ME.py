import torch
import torch.nn.functional as F
import numpy as np
import os
import portalocker
import glob
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import cupy as cp
from tqdm import tqdm
from utils import read_pickle, write_pickle
from swd import add_noise


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


def add_eps_to_hist(hist, percent=1):
    eps_end = (percent / 100) * (1 / hist.numel())
    eps_start = eps_end / (1 - hist.numel()*eps_end)
    hist += eps_start
    hist /= hist.sum()
    return hist


def new_kl_divergence(p, q):
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
