import cupy as cp
import math

def determine_padding(filter_shape, output_shape="same"):

    if output_shape == "valid":
        return (0, 0), (0, 0)
    
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)

# Reference: CS231n Stanford
def get_im2col_indices(images_shape, filter_shape, padding, stride=1):

    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + pad_h[0]+pad_h[1] - filter_height) / stride + 1)
    out_width = int((width + pad_w[0]+pad_w[1] - filter_width) / stride + 1)

    i0 = cp.repeat(cp.arange(filter_height), filter_width)
    i0 = cp.tile(i0, channels)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(filter_width), filter_height * channels)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    i = cp.reshape(i0, (-1, 1)) + cp.reshape(i1, (1, -1))
    j = cp.reshape(j0, (-1, 1)) + cp.reshape(j1, (1, -1))

    k = cp.reshape(cp.repeat(cp.arange(channels), filter_height * filter_width), (-1, 1))

    return (k, i, j)


# Reference: CS231n Stanford
def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape

    pad_h, pad_w = determine_padding(filter_shape, output_shape)

    images_padded = cp.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    cols = images_padded[:, k, i, j]
    channels = images.shape[1]

    cols = cp.reshape(cp.transpose(cols, (1, 2, 0)),(filter_height * filter_width * channels, -1))
    return cols


# Reference: CS231n Stanford
def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + pad_h[0]+pad_h[1]
    width_padded = width + pad_w[0]+pad_w[1]
    images_padded = cp.zeros((batch_size, channels, height_padded, width_padded))

    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    cols = cp.reshape(cols, (channels * filter_shape[0]*filter_shape[1], -1, batch_size))
    cols = cp.transpose(cols, (2, 0, 1))

    cp.scatter_add(images_padded, (slice(None), k, i, j), cols)

    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]