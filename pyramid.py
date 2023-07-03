from math import floor, sqrt, log2, ceil
from ResizeRight.resize_right import resize


def pyramid_shapes_scale_2(h, w, patch_size=5, min_dim_size=40):
    assert min_dim_size < min(h, w)

    # current # of patches
    edge_len = patch_size - 1
    h -= edge_len
    w -= edge_len

    # calc num_scales
    num_scales = floor(log2(min(h, w) / min_dim_size)) + 1

    # desired # of patches
    smallest_scale = 2 ** (num_scales - 1)
    h = smallest_scale * (h // smallest_scale)
    w = smallest_scale * (w // smallest_scale)

    return [(int(h/(2**i)) + edge_len, int(w/(2**i)) + edge_len,) for i in range(num_scales)]


def pyramid_shapes_scale_sqrt_2(h, w, patch_size=5, min_dim_size=40):
    even_scales = pyramid_shapes_scale_2(h=h, w=w, patch_size=patch_size, min_dim_size=min_dim_size)
    odd_scales = pyramid_shapes_scale_2(h=floor(h/sqrt(2)), w=floor(w/sqrt(2)), patch_size=patch_size, min_dim_size=min_dim_size)

    num_scales = len(even_scales) + len(odd_scales)
    if num_scales % 2 == 0:
        return [val for pair in zip(even_scales, odd_scales) for val in pair]

    odd_scales.append(None)
    return [val for pair in zip(even_scales, odd_scales) for val in pair][:-1]


def remove_borderd(image_pyramid, patch_size):
    half_patch_size = patch_size / 2

    # calculate center patch location at lowest resolution image
    h, w = image_pyramid[-1].shape[2:]
    init_h_percent, end_h_percent = half_patch_size / h, (h - half_patch_size) / h
    init_w_percent, end_w_percent = half_patch_size / w, (w - half_patch_size) / w

    # crop relevant border from each image
    new_image_pyramid = []
    for image in image_pyramid:
        h, w = image.shape[2:]
        init_h, end_h = floor((init_h_percent * h) - half_patch_size), ceil((end_h_percent * h) + half_patch_size)
        init_w, end_w = floor((init_w_percent * w) - half_patch_size), ceil((end_w_percent * w) + half_patch_size)
        new_image = image[:, :, init_h:end_h, init_w:end_w]
        new_image_pyramid.append(new_image)

    return new_image_pyramid


def generate_image_pyramid(image, min_dim_size):
    h, w = image.shape[2:]
    num_scales = floor(log2(min(h, w) / min_dim_size) * 2) + 1
    image_pyramid = []
    for i in range(num_scales):
        scale_factor = 1 / (sqrt(2) ** i)
        out_h, out_w = round(h * scale_factor), round(w * scale_factor)
        resized_image = resize(image, out_shape=(out_h, out_w), pad_mode='reflect').clip(0, 1)
        image_pyramid.append(resized_image)

    return image_pyramid
