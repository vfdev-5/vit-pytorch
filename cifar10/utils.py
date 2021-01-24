import os

import torch
from torchvision import models
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Pad, RandomCrop, RandomHorizontalFlip

from vit import VisionTransformer


train_transform = Compose(
    [
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])


def get_train_test_datasets(path):
    if not os.path.exists(path):
        os.makedirs(path)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    train_ds = datasets.CIFAR10(root=path, train=True, download=download, transform=train_transform)
    test_ds = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)

    return train_ds, test_ds


def get_model(name):
    __dict__ = globals()

    torch_scripted = False
    if name.startswith("torchscripted_"):
        torch_scripted = True
        name = name[len("torchscripted_"):]

    if name in models.__dict__:
        fn = models.__dict__[name]
    elif name in ["vit_tiny_patch4_32x32", "vit_tiny_patch2_32x32", "vit_b4_32x32", "vit_b3_32x32", "vit_b2_32x32"]:
        fn = __dict__[name]
    elif name in ["timm_vit_b4_32x32", ]:
        try:
            import timm
        except ImportError:
            raise RuntimeError(
                "Package timm is not installed. Please, install it with:\n"
                "\tpip install timm"
            )
        fn = __dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    model = fn(num_classes=10)
    if torch_scripted:
        model = torch.jit.script(model)

    return model


def vit_tiny_patchX_32x32(patch_size, num_classes=10, input_channels=3):
    return VisionTransformer(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=32,
        patch_size=patch_size,
        hidden_size=512,
        num_layers=4,
        num_heads=6,
        mlp_dim=1024,
        drop_rate=0.1, 
        attn_drop_rate=0.0,
    )


def vit_tiny_patch4_32x32(num_classes=10, input_channels=3):
    return vit_tiny_patchX_32x32(4, num_classes=num_classes, input_channels=input_channels)


def vit_tiny_patch2_32x32(num_classes=10, input_channels=3):
    return vit_tiny_patchX_32x32(2, num_classes=num_classes, input_channels=input_channels)


def vit_b4_32x32(num_classes=10, input_channels=3):
    return VisionTransformer(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=32,
        patch_size=4,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    )


def vit_b3_32x32(num_classes=10, input_channels=3):
    return VisionTransformer(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=32,
        patch_size=3,  # ceil of 32 / (224 / 16) = 2.286
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    )


def vit_b2_32x32(num_classes=10, input_channels=3):
    return VisionTransformer(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=32,
        patch_size=2,  # floor of 32 / (224 / 16) = 2.286
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    )


def timm_vit_b4_32x32(num_classes=10, input_channels=3):
    from functools import partial
    import torch.nn as nn
    from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

    return TimmVisionTransformer(
        img_size=32, patch_size=4, in_chans=input_channels,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        num_classes=num_classes, norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
