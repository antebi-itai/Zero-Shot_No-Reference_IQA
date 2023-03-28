from math import floor, ceil, sqrt


def pyramid_scale_2_shapes(image, num_scales=4, patch_size=5, dim_max_size=1024):
    h, w = image.shape[2:]
    assert max(h, w) >= dim_max_size, "image too small"

    # scale so max is dim_max_size
    multiplier = dim_max_size / max(h, w)
    h, w = floor(h * multiplier), floor(w * multiplier)

    # current # of patches
    edge_len = patch_size - 1
    h -= edge_len
    w -= edge_len

    # desired # of patches
    smallest_scale = 2 ** (num_scales - 1)
    h = smallest_scale * (h // smallest_scale)
    w = smallest_scale * (w // smallest_scale)

    return [((h/(2**i)) + edge_len, (w/(2**i)) + edge_len,) for i in range(num_scales)]


def pyramid_scale_sqrt_2_shapes(image, num_scales=9, patch_size=5, dim_max_size=1024):
    even_scales = pyramid_scale_2_shapes(image=image, num_scales=ceil(num_scales/2), patch_size=patch_size, dim_max_size=dim_max_size)
    odd_scales = pyramid_scale_2_shapes(image=image, num_scales=floor(num_scales/2), patch_size=patch_size, dim_max_size=floor(dim_max_size/sqrt(2)))

    if num_scales % 2 == 0:
        return [val for pair in zip(even_scales, odd_scales) for val in pair]

    odd_scales.append(None)
    return [val for pair in zip(even_scales, odd_scales) for val in pair][:-1]
