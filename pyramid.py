from math import floor, log2
from ResizeRight.resize_right import resize


def create_image_pyramids(image, patch_size=5, min_dim_size=40):
    h, w = image.shape[-2:]
    assert min_dim_size < min(h, w)

    # calc number of  scales in image pyramid
    num_scales = floor(log2(min(h, w) / min_dim_size)) + 1

    # desired highest level shape
    max_shrink_scale = 2 ** (num_scales - 1)
    h = max_shrink_scale * (h // max_shrink_scale)
    w = max_shrink_scale * (w // max_shrink_scale)

    # full_size_image pyramid
    full_size_image_pyramid_shapes = [(int(h/(2**i)), int(w/(2**i)),) for i in range(num_scales)]
    full_size_image_pyramid = [resize(image, out_shape=image_shape, pad_mode='reflect').clip(0, 1) for image_shape in full_size_image_pyramid_shapes]

    # residual pixels to remove from full_size_image
    border_width = patch_size // 2
    pixels_outside_image_pyramid = [border_width**(num_scales - i) for i in range(num_scales)]
    pixels_outside_image_with_borders_pyramid = [(pixels_outside_image - border_width) for pixels_outside_image in pixels_outside_image_pyramid]

    # crop full_size_image to obtain image and image_with_borders
    image_pyramid = []
    image_pyramid_with_borders = []
    for full_size_image, pixels_outside_image, pixels_outside_image_with_borders in \
            zip(full_size_image_pyramid, pixels_outside_image_pyramid, pixels_outside_image_with_borders_pyramid):
        # image
        start, end = (pixels_outside_image, -pixels_outside_image) if pixels_outside_image != 0 else (None, None)
        image_pyramid.append(full_size_image[..., start:end, start:end])
        # image_with_borders
        start, end = (pixels_outside_image_with_borders, -pixels_outside_image_with_borders) if pixels_outside_image_with_borders != 0 else (None, None)
        image_pyramid_with_borders.append(full_size_image[..., start:end, start:end])
    return image_pyramid, image_pyramid_with_borders, full_size_image_pyramid
