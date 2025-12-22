def conv2d_size_out(
    size: int,
    kernel_size: int,
    stride: int
) -> int:
    return (size - (kernel_size - 1) - 1) // stride + 1


def conv_transpose2d_size_out(
    size: int,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    output_padding: int = 0,
) -> int:
    return (size - 1) * stride - 2 * padding + kernel_size + output_padding
