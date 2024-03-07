"""This module handles registering prithvi_swin models into timm.
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import torch
from terratorch.datasets.utils import HLSBands
from terratorch.models.backbones.prithvi_select_patch_embed_weights import (
    prithvi_select_patch_embed_weights,
)
from timm.models._builder import build_model_with_cfg
from timm.models._registry import generate_default_cfgs, register_model

from .swin3d import SwinTransformer3D

PRETRAINED_BANDS: list[HLSBands | int] = [
    HLSBands.BLUE,
    HLSBands.GREEN,
    HLSBands.RED,
    HLSBands.NIR_NARROW,
    HLSBands.SWIR_1,
    HLSBands.SWIR_2,
]


def _cfg(file: Path = "", **kwargs) -> dict:
    return {
        "file": file,
        "source": "file",
        "license": "mit",
        # "first_conv": "patch_embed.proj",
        **kwargs,
    }


# default_cfgs = generate_default_cfgs(
#     {
#         # add your pretrained weights here and uncomment
#         # "prithvi_swin_3d": _cfg(
#         #     file="/dccstor/geofm-finetuning/pretrain_ckpts/swin_weights/2023-07-24_14-06-22/epoch-99-loss-0.1632_mmseg.pt"
#         # ),
#     }
# )


def convert_weights_swin2mmseg(ckpt):
    # from https://github.com/open-mmlab/mmsegmentation/blob/main/tools/model_converters/swin2mmseg.py
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith("head"):
            continue
        elif k.startswith("layers"):
            new_v = v
            if "attn." in k:
                new_k = k.replace("attn.", "attn.w_msa.")
            elif "mlp." in k:
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "ffn.layers.0.0.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "ffn.layers.1.")
                else:
                    new_k = k.replace("mlp.", "ffn.")
            elif "downsample" in k:
                new_k = k
                if "reduction." in k:
                    new_v = correct_unfold_reduction_order(v)
                elif "norm." in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace("layers", "stages", 1)
        elif k.startswith("patch_embed"):
            new_v = v
            if "proj" in k:
                new_k = k.replace("proj", "projection")
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt[new_k] = new_v

    return new_ckpt


# If you need to adapt the checkpoint file, do it here
# def checkpoint_filter_fn(
#     state_dict: dict[str, torch.Tensor],
#     model: torch.nn.Module,
#     pretrained_bands,
#     model_bands,
# ):
#
#     return state_dict


def _create_swin_3D(
    variant: str,
    pretrained_bands: list[HLSBands | int],
    model_bands: Sequence[HLSBands | int],
    pretrained: bool = False,  # noqa: FBT002, FBT001
    **kwargs,
):
    # what layer indices should be output by default
    default_out_indices = tuple(
        i for i, _ in enumerate(kwargs.get("depths", (1, 1, 3, 1)))
    )
    out_indices = kwargs.pop("out_indices", default_out_indices)

    # the swin model does not take this kwarg
    kwargs_filter = ("num_frames", "num_classes")
    kwargs["in_chans"] = len(model_bands)

    # If you need to adapt the checkpoint file
    # def checkpoint_filter_wrapper_fn(state_dict, model):
    #     return checkpoint_filter_fn(state_dict, model, pretrained_bands, model_bands)

    model: torch.nn.Module = build_model_with_cfg(
        SwinTransformer3D,
        variant,
        pretrained,
        # if you need to adapt the checkpoint file
        # pretrained_filter_fn=checkpoint_filter_wrapper_fn,
        pretrained_strict=False,
        feature_cfg={
            "flatten_sequential": True,
            "out_indices": out_indices,
        },
        kwargs_filter=kwargs_filter,
        **kwargs,
    )
    model.pretrained_bands = pretrained_bands
    model.model_bands = model_bands

    # how should the features be processed before passing to the decoder
    def prepare_features_for_image_model(x):
        return [layer_output.squeeze(2).contiguous() for layer_output in x]

    # add permuting here
    model.prepare_features_for_image_model = prepare_features_for_image_model
    return model


@register_model
def prithvi_swin_3d(
    pretrained: bool = False,  # noqa: FBT002, FBT001
    pretrained_bands: list[HLSBands | int] | None = None,
    bands: list[HLSBands | int] | None = None,
    **kwargs,
) -> torch.nn.Module:
    """Prithvi Swin 3D"""
    if pretrained_bands is None:
        pretrained_bands = PRETRAINED_BANDS
    if bands is None:
        bands = pretrained_bands
        logging.info(
            f"Model bands not passed. Assuming bands are ordered in the same way as {PRETRAINED_BANDS}.\
            Pretrained patch_embed layer may be misaligned with current bands"
        )

    model_args = {
        "patch_size": (4, 4, 4),
        "window_size": (2, 7, 7),
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "in_chans": 6,
        "num_heads": (3, 16, 12, 24),
    }
    transformer = _create_swin_3D(
        "prithvi_swin_3d",
        pretrained_bands,
        bands,
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )
    return transformer
