from urllib.request import urlopen

import timm
import torch
from PIL import Image
from timm.layers import PatchEmbed as TimmPatchEmbed
from timm.models import VisionTransformer
from timm.data import resolve_model_data_config

from nest import PatchEmbed

if __name__ == '__main__':
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 32

    timm_patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten=False,
    )

    patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        flatten=False,
    )

    model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    img = Image.open(urlopen(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))
    transforms = timm.data.create_transform(**data_config, is_training=False)
    x = transforms(img).unsqueeze(0)  # (B, C, H, W)

    with torch.no_grad():
        output = model(x)

        y = timm_patch_embed(x)
        z = patch_embed(x)

        print(output.shape, y.shape, z.shape)
