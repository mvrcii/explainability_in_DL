import logging

import timm
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------
# This script is based on and inspired by code from the "mildlyoverfitted" repository.
# Original Source: https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py
# The code from this repository has been modified and used as a reference for certain functionalities in this script.
# --------------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).

    patch_size : int
        Size of the patch (it is a square).

    in_chans : int
        Number of input channels.

    embed_dim : int
        The embedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches
        and their embedding.
    """

    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Parameters
        __________
        x : torch.Tensor
            Shape `(batch_size, in_chans, img_size, img_size)`.

        Returns
        _______
        torch.Tensor
            Shape `(batch_size, n_patches, embed_dim)`.
        """
        x = self.proj(x)  # (batch_size, embed_dims, n_patches ** 0.5, n_patches ** 0.5)
        x = x.flatten(2)  # (batch_size, embed_dims, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)

        return x


class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    embed_dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing constant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, embed_dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads  # concatenated heads will later have same dim as our input
        self.scale = self.head_dim ** -0.5  # don't feed extremely large values into softmax which causes small gradients

        self.qkv = nn.Linear(embed_dim, embed_dim * 3,
                             bias=qkv_bias)  # can also be written with 3 separate linear mappings
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(embed_dim, embed_dim)  # takes concatenated heads and maps them into a new space
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, embed_dim)`.
            Here, `n_patches + 1` includes an additional class token.

        Returns
        -------
        torch.Tensor
            Shape `(batch_size, n_patches + 1, embed_dim)`.
            Here, `n_patches + 1` includes an additional class token.
        """
        batch_size, n_patches, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError("Embedding Dimension of the Input Tensor must match the Configs Embedding Dimension!")

        # Linear projection to get combined QKV tensor
        qkv = self.qkv(x)  # (batch_size, n_patches + 1, 3 * embed_dim)

        # Splitting QKV into separate tensors
        qkv = qkv.reshape(batch_size, n_patches, 3, self.n_heads, self.head_dim)  # (batch_size, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv.unbind(dim=0)  # Unbind along QKV dimension (batch_size, n_patches + 1, n_heads, head_dim)

        # If CUDA available scaled_dot_product_attention yields better performance
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = attn.softmax(dim=-1)  # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (batch_size, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (batch_size, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (batch_size, n_patches + 1, embed_dim)

        x = self.proj(weighted_avg)  # (batch_size, n_patches + 1, embed_dim)
        x = self.proj_drop(x)  # (batch_size, n_patches + 1, embed_dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    dropout_p : float
        Dropout probability.

    Attributes
    ----------
    fc1 : nn.Linear
        The First linear layer.

    activation_fn : nn.GELU
        GELU activation function: https://paperswithcode.com/method/gelu.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """

    def __init__(self, in_features, hidden_features, out_features, dropout_p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation_fn = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(batch_size, n_patches +1, out_features)`
        """
        x = self.fc1(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.activation_fn(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.drop(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (batch_size, n_patches + 1, out_features)
        x = self.drop(x)  # (batch_size, n_patches + 1, out_features)

        return x


class Block(nn.Module):
    """Transformer block.

       Parameters
       ----------
       embed_dim : int
           Embedding dimension.

       n_heads : int
           Number of attention heads.

       mlp_ratio : float
           Determines the hidden dimension size of the `MLP` module with respect
           to `embed_dim`.

       qkv_bias : bool
           If True then we include bias to the query, key and value projections.

       p, attn_p : float
           Dropout probabilities.

       Attributes
       ----------
       norm1, norm2 : LayerNorm
           Layer normalization.

       attn : Attention
           Attention module.

       head : MLP Classifier Head
           MLP module.
       """

    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = Attention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        # Final normalization layer
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        # Classifier
        self.head = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout_p=p
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.head(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision Transformer.

    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).

    patch_size : int
        Both height and the width of the patch (it is a square).

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes of the classification problem.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of transformer blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `embed_dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probabilities.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls_token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """

    def __init__(self,
                 img_size=384,
                 patch_size=16,
                 in_chans=3,
                 n_classes=1000,
                 embed_dim=768,
                 depth=12,
                 n_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 p=0.,
                 attn_p=0.):
        super().__init__()

        # First Layer - Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        # Create the CLS token which will always be prepended to the patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Create the positional embeddings. Its goal is to determine the position of each patch.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        # Create the Transformer Encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, in_chans, img_size, img_size)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(batch_size, n_classes)`.
        """
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        # Take the class token and replicate it over the batch dimension
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)

        # Prepend the class token to the patch embeddings
        x = torch.cat((cls_token, x), dim=1)  # (batch_size, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (batch_size, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        # Iteratively define all the blocks of our Transformer Encoder
        for block in self.blocks:
            x = block(x)

        # Layer Normalization
        x = self.norm(x)

        # Extract the CLS token and pass it through the MLP head for classification
        # We are hoping that this embedding encodes the meaning of the entire image
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x

