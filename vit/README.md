# Vision Transformer (ViT)

This is a PyTorch implementation of the Vision Transformer (ViT) model from the
paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by
Alexey Dosovitskiy et al.

ViT is a simple way to apply the Transformer architecture to image classification. It works by splitting the image into 
patches and then flattening them into sequences of vectors. The Transformer encoder is then applied to these sequences 
of vectors, producing a sequence of vectors that can be used as input to a linear classifier.

## Pretrained Weights
In order to use the pretrained weights from timm, first call `download_weights.py` from the root. This will download the
weights and save them in the `vit` folder. The weights can then be loaded using `torch.load`.

## Inference
Call `inference.py` from the root to run inference on an image.

## Verify the model
The script `verify.py` can be used to verify the custom model against the pretrained model implementation from timm.
