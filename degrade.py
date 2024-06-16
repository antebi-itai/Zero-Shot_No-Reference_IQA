import torch
import torch.nn.functional as F
import numpy as np
import cv2
import skimage as sk
from PIL import Image
from io import BytesIO
from images import pt2np, np2pt
from IQA.utils import pt2im, im2pt


def glass_blur(image, sigma=10, max_delta=10, iterations=1):
    _, _, image_h, image_w = image.shape
    orig_h, orig_w = torch.meshgrid(torch.arange(0, image_h), torch.arange(0, image_w))

    # blur
    image = np2pt(sk.filters.gaussian(pt2np(image), sigma=sigma, channel_axis=-1)).clip(0, 1)

    # locally shuffle pixels
    for i in range(iterations):
        dh, dw = torch.randint(-max_delta, max_delta + 1, size=(2, image_h, image_w))
        h_prime_map, w_prime_map = orig_h + dh, orig_w + dw
        for h in range(max_delta, image_h - max_delta):
            for w in range(max_delta, image_w - max_delta):
                # swap
                h_prime, w_prime = h_prime_map[h, w], w_prime_map[h, w]
                image[..., h, w], image[..., h_prime, w_prime] = image[..., h_prime, w_prime], image[..., h, w]

    # blur
    image = np2pt(sk.filters.gaussian(pt2np(image), sigma=sigma, channel_axis=-1)).clip(0, 1)
    return image


def multivariate_gaussian_kernel(shape, angle, sigma_x, sigma_y):
    height, width = shape
    kernel = np.zeros(shape)

    cos_theta = np.cos(np.radians(angle))
    sin_theta = np.sin(np.radians(angle))

    for x in range(-width // 2, width // 2):
        for y in range(-height // 2, height // 2):
            x_rot = cos_theta * x - sin_theta * y
            y_rot = sin_theta * x + cos_theta * y
            kernel[y + height // 2, x + width // 2] = np.exp(
                -(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2))
            )

    return kernel / np.sum(kernel)


def jpeg_compress(image, quality=75):
    assert 0 <= quality <= 95, "JPEG quality scales from 0 (worst) to 95 (best)"
    image = pt2im(image)
    buffer = BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    image = Image.open(buffer)
    image = im2pt(image)
    return image


def degrade(reference, degradation, severity):
    if (severity == 0) or (degradation == "reference"):
        output = reference

    else:
        # blur
        if degradation == "gauss_blur":
            sigma = 3 * severity
            output = np2pt(sk.filters.gaussian(pt2np(reference), sigma=sigma, channel_axis=-1))
        elif degradation == "glass_blur":
            sigma, max_delta, iterations = [(2, 2, 2), (3, 3, 3), (4, 4, 5), (5, 5, 6), (6, 6, 8), (7, 7, 9), (8, 8, 11)][severity - 1]
            output = glass_blur(reference, sigma=sigma, max_delta=max_delta, iterations=iterations)
        elif degradation == "motion_blur":
            kernel_size = 19 * severity
            kernel = multivariate_gaussian_kernel(shape=(kernel_size, kernel_size), angle=45, sigma_x=kernel_size/3, sigma_y=kernel_size/15)
            output = np2pt(cv2.filter2D(pt2np(reference), -1, kernel))
        # noise
        elif degradation == "white_noise":
            std = 0.07 * severity
            output = (reference + torch.normal(mean=torch.zeros_like(reference), std=std))
        elif degradation == "speckle_noise":
            var = (0.07 * 2 * severity) ** 2
            output = np2pt(sk.util.random_noise(pt2np(reference), mode='speckle', var=var).astype(np.float32))
        elif degradation == "impulse_noise":
            amount = [.03, .06, .10, 0.15, 0.20, 0.25, 0.30][severity - 1]
            output = np2pt(sk.util.random_noise(pt2np(reference), mode='s&p', amount=amount))
        # other
        elif degradation == "pixelate_avg":
            kernel_size = 19 * severity
            output = F.interpolate(F.avg_pool2d(reference, kernel_size=kernel_size), scale_factor=kernel_size)
        elif degradation == "pixelate_delta":
            kernel_size = 19 * severity
            output = F.interpolate(reference[..., ::kernel_size, ::kernel_size], scale_factor=kernel_size)
        elif degradation == "jpeg":
            quality = [95, 50, 25, 15, 10, 7, 4][severity - 1]
            output = jpeg_compress(reference, quality=quality)
        else:
            raise NotImplementedError

    output = (output * 255).round() / 255
    output = output.clip(0, 1).to(device=reference.device)
    return output
