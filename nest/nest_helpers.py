import math


def blockify(image, block_size: int):
    """image to blocks
    Args:
        image (Tensor): with shape (B, H, W, C)
        block_size (int): edge length of a single square block in units of H, W
    """
    B, H, W, C = image.shape
    assert H % block_size == 0, '`block_size` must divide input height evenly'
    assert W % block_size == 0, '`block_size` must divide input width evenly'
    grid_height = H // block_size
    grid_width = W // block_size
    blocks = image.reshape(B, grid_height, block_size, grid_width, block_size, C)  # (B, grid_height, block_size, grid_width, block_size, C)
    blocks = blocks.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    return blocks  # (B, T, N, C) with T = number of blocks, N = sequence size per block


def deblockify(blocks, block_size: int):
    """blocks to image
    Args:
        blocks (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired H, W
    """
    B, T, _, C = blocks.shape
    grid_size = int(math.sqrt(T))
    height = width = grid_size * block_size
    image = blocks.reshape(B, grid_size, grid_size, block_size, block_size, C)
    image = image.transpose(2, 3).reshape(B, height, width, C)
    return image  # (B, H, W, C)
