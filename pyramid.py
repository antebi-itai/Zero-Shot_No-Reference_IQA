from math import floor, sqrt, log2


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
