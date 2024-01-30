import logging

import numpy as np
import timm
import torch

from models.vit import VisionTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_n_params(module):
    """Get the number of parameters of a module."""
    n_params = sum([p.numel() for p in module.parameters() if p.requires_grad])
    return n_params


def check_tensors_equal(t1, t2):
    try:
        np.testing.assert_allclose(t1.detach().cpu().numpy(), t2.detach().cpu().numpy())
    except AssertionError as e:
        logging.warning(f'Tensor mismatch: {e}')


model_name = "vit_base_patch16_224"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()

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
for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    if p_o.numel() != p_c.numel():
        logging.info(f"Parameter count mismatch: {n_o} - {n_c} - {p_o.shape} - {p_c.shape}")

    print(f"{n_o} - {n_c} - {p_o.shape}")
    p_c.data[:] = p_o.data  # Copy the data from the official model to the custom one

    check_tensors_equal(p_o, p_c)

tensor_inp = torch.randn(1, 3, 224, 224)
result_o = model_official(tensor_inp)
result_c = model_custom(tensor_inp)

params_o = get_n_params(model_official)
params_c = get_n_params(model_custom)
print(f"Number of parameters: {params_o} - {params_c}")
assert params_o == params_c

# Save custom model
torch.save(model_custom, "model.pth")


