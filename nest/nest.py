import collections.abc
import math
from functools import partial

import torch
import torch.nn as nn
from timm.layers import create_conv2d, create_pool2d, to_ntuple, to_2tuple, create_classifier, trunc_normal_, Mlp as MLP
from timm.models import named_apply, register_model, build_model_with_cfg
from timm.models.nest import _init_nest_weights
from torchviz import make_dot

from nest_helpers import deblockify, blockify


# --------------------------------------------------------------------------------
# This script is based on and inspired by code from the "mildlyoverfitted" repository.
# Original Source: https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py
# The code from this repository has been modified and used as a reference for certain functionalities in this script.
# --------------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True, flatten=True):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.flatten = flatten

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"Input height ({H}) and width ({W}) must be divisible by the patch size ({self.patch_size})."

        x = self.proj(x)  # (B, C, H, W) -> (B, C, H//p, W//p)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, C, H//p, W//p) -> (B, N, C) where N = H*W/p^2

        return x


class Attention(nn.Module):
    """
        This is much like ViT's Attention, but uses *localized* self-attention by accepting an input with
        an extra "image block" dimension
    """

    def __init__(self, embed_dim, num_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # don't feed extremely large values into softmax which causes small gradients

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)  # takes concatenated heads and maps them into a new space
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        x is shape: B (Batch Size), T (Image Blocks), N (Sequence Length / Number of Patches), C (Embedding Dimension)
            note: H (Number of Heads), C' (Head Dimension)
        """
        B, T, N, C = x.shape

        if C != self.embed_dim:
            raise ValueError("Embedding Dimension of the Input Tensor must match the models Embedding Dimension!")

        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, self.head_dim)  # (B, T, N, 3, H, C')

        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # (qkv, B, H, T, N, C')
        q, k, v = qkv.unbind(dim=0)  # 3x (B, H, T, N, C')

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B, H, T, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Last two dimensions determine result dim (N,N) x (N,C') = (N, C') -> (_, _, _, N, N) @ (_, _, _, N, C')
        x = attn @ v  # (B, H, T, N, C')
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, N, C', H)
        x = x.reshape(B, T, N, C)  # (B, T, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class TransformerLayer(nn.Module):
    """
    Identical to ViT Transformer Block
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, proj_p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=proj_p
        )
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embed_dim,
            act_layer=nn.GELU,
            drop=proj_p
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, pad_type=''):
        super().__init__()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        assert x.shape[-2] % 2 == 0, 'BlockAggregation requires even input spatial dims'
        assert x.shape[-1] % 2 == 0, 'BlockAggregation requires even input spatial dims'
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)


class NestLevel(nn.Module):
    """One single hierarchical level/layer of a Nested Transformer."""

    def __init__(
            self,
            num_blocks,
            block_size,
            seq_length,
            depth,
            embed_dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            proj_p=0.,
            attn_p=0.,
            pad_type='',
            prev_embed_dim=None,
    ):
        super().__init__()
        self.block_size = block_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_blocks, seq_length, embed_dim))  # (1, T, N, C)

        if prev_embed_dim is not None:
            self.convPool = ConvPool(prev_embed_dim, embed_dim, pad_type=pad_type)
        else:
            self.convPool = nn.Identity()

        self.transformer_encoder = nn.Sequential(*[
            TransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=proj_p
            ) for _ in range(depth)
        ])

    def forward(self, x):
        """
        x shape: (B, C, N, N)
        """
        x = self.convPool(x)  # (B, C, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer

        x = blockify(x, self.block_size)  # (B, T, N, C')
        x = x + self.pos_embed

        x = self.transformer_encoder(x)  # (B, T, N, C')
        x = deblockify(x, self.block_size)  # (B, H', W', C')

        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 3, 1, 2)  # (B, C, H', W')


class Nest(nn.Module):
    """ Nested Transformer (NesT)

    A PyTorch impl of : `Aggregating Nested Transformers`
        - https://arxiv.org/abs/2105.12723
    """

    def __init__(
            self,
            img_size=224,
            in_chans=3,
            patch_size=4,
            num_levels=3,
            embed_dims=(128, 256, 512),
            num_heads=(4, 8, 16),
            depths=(2, 2, 20),
            num_classes=1000,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.5,
            # norm_layer=None,
            # act_layer=None,
            pad_type='',
            weight_init='',
            global_pool='avg',
    ):
        """
        Args:
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            patch_size (int): patch size
            num_levels (int): number of block hierarchies (T_d in the paper)
            embed_dims (int, tuple): embedding dimensions of each level
            num_heads (int, tuple): number of attention heads for each level
            depths (int, tuple): number of transformer layers for each level
            num_classes (int): number of classes for classification head
            mlp_ratio (int): ratio of mlp hidden embed_dim to embedding embed_dim for MLP of transformer layers
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate for MLP of transformer layers, MSA final projection layer, and classifier
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer for transformer layers
            act_layer: (nn.Module): activation layer in MLP of transformer layers
            pad_type: str: Type of padding to use '' for PyTorch symmetric, 'same' for TF SAME
            weight_init: (str): weight init scheme
            global_pool: (str): type of pooling operation to apply to final feature map

        Notes:
            - Default values follow NesT-B from the original Jax code.
            - `embed_dims`, `num_heads`, `depths` should be ints or tuples with length `num_levels`.
            - For those following the paper, Table A1 may have errors!
                - https://github.com/google-research/nested-transformer/issues/2
        """
        super().__init__()

        for param_name in ['embed_dims', 'num_heads', 'depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == num_levels, f'Require `len({param_name}) == num_levels`'

        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        self.feature_info = []
        # norm_layer = norm_layer or LayerNorm
        # act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        self.num_levels = num_levels

        assert img_size % patch_size == 0, '`patch_size` must divide `img_size` evenly'
        self.patch_size = patch_size

        # Number of blocks at each level
        self.num_blocks = (4 ** torch.arange(num_levels)).flip(0).tolist()
        assert (img_size // patch_size) % math.sqrt(self.num_blocks[0]) == 0, \
            'First level blocks don\'t fit evenly. Check `img_size`, `patch_size`, and `num_levels`'

        # Block edge size in units of patches
        # Hint: (img_size // patch_size) gives number of patches along edge of image. sqrt(self.num_blocks[0]) is the
        #  number of blocks along edge of image
        self.block_size = int((img_size // patch_size) // math.sqrt(self.num_blocks[0]))

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            flatten=False,
        )
        self.num_patches = self.patch_embed.num_patches
        self.seq_length = self.num_patches // self.num_blocks[0]

        # Build up each hierarchical level
        levels = []
        # dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        curr_stride = 4

        for i in range(len(self.num_blocks)):
            embed_dim = embed_dims[i]
            levels.append(NestLevel(
                num_blocks=self.num_blocks[i],
                block_size=self.block_size,
                seq_length=self.seq_length,
                depth=depths[i],
                embed_dim=embed_dim,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                prev_embed_dim=prev_dim,
                proj_p=proj_drop_rate,
                attn_p=attn_drop_rate,
                # drop_path=dp_rates[i],
                # norm_layer=norm_layer,
                # act_layer=act_layer,
                pad_type=pad_type,
            ))
            self.feature_info += [dict(num_chs=embed_dim, reduction=curr_stride, module=f'levels.{i}')]
            prev_dim = embed_dim
            curr_stride *= 2
        self.levels = nn.Sequential(*levels)

        # Final normalization layer
        self.norm = nn.LayerNorm(embed_dims[-1])

        # Classifier
        global_pool, head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)
        self.global_pool = global_pool
        self.head_drop = nn.Dropout(drop_rate)
        self.head = head

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        for level in self.levels:
            trunc_normal_(level.pos_embed, std=.02, a=-2, b=2)
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.head = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.levels(x)
        # Layer norm done over channel dim only (to NHWC and back)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


if __name__ == '__main__':
    model = Nest(
        img_size=224,
        in_chans=3,
        patch_size=14,
        num_levels=3,
        embed_dims=(32, 32, 32),
        num_heads=(4, 8, 16),
        depths=(2, 2, 20),
        num_classes=1000,
    )
    model.eval()
    print(model)

    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        print(y)
