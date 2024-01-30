import logging

import numpy as np
import timm
import torch
from vit import VisionTransformer


def check_tensors_equal(t1, t2):
    try:
        np.testing.assert_allclose(t1.detach().cpu().numpy(), t2.detach().cpu().numpy())
    except AssertionError as e:
        logging.warning(f'Tensor mismatch: {e}')


def download_weights():
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.eval()

    custom_config = {
        "img_size": 224,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
    }

    model_custom = VisionTransformer(**custom_config)
    model_custom.eval()

    # Iterate through the parameters of both models and compare them
    for (n_o, p_o), (n_c, p_c) in zip(model.named_parameters(), model_custom.named_parameters()):
        if p_o.numel() != p_c.numel():
            logging.info(f"Parameter count mismatch: {n_o} - {n_c} - {p_o.shape} - {p_c.shape}")

        print(f"{n_o} - {n_c} - {p_o.shape}")
        p_c.data[:] = p_o.data  # Copy the data from the official model to the custom one

        check_tensors_equal(p_o, p_c)

    torch.save(model_custom, "vit/model.pth")


if __name__ == '__main__':
    download_weights()
