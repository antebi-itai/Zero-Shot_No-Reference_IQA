import torch
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
    if severity == 0:
        output = reference

    else:
        if degradation == "reference":
            output = reference
        elif degradation == "gauss_blur":
            sigma = 3 * severity
            output = np2pt(sk.filters.gaussian(pt2np(reference), sigma=sigma, channel_axis=-1))
        elif degradation == "glass_blur":
            sigma, max_delta, iterations = [(2, 1, 2), (2, 2, 3), (3, 3, 4), (3, 4, 5), (4, 5, 5), (4, 6, 5), (5, 7, 5)][severity - 1]
            output = glass_blur(reference, sigma=sigma, max_delta=max_delta, iterations=iterations)
        elif degradation == "impulse_noise":
            amount = [.03, .06, .10, 0.15, 0.20, 0.25, 0.30][severity - 1]
            output = np2pt(sk.util.random_noise(pt2np(reference), mode='s&p', amount=amount))
        elif degradation == "jpeg":
            quality = [95, 50, 25, 15, 10, 7, 4][severity - 1]
            output = jpeg_compress(reference, quality=quality)
        elif degradation == "white_noise":
            std = 0.07 * severity
            output = (reference + torch.normal(mean=torch.zeros_like(reference), std=std))
        else:
            raise NotImplementedError

    output = (output * 255).round() / 255
    output = output.clip(0, 1).to(device=reference.device)
    return output
