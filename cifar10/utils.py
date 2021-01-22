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
    elif name in ["vit_tiny", "vit_b16_32x32"]:
        fn = __dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    model = fn(num_classes=10)
    if torch_scripted:
        model = torch.jit.script(model)

    return model


def vit_tiny(num_classes=10, input_channels=3, input_size=32):
    return VisionTransformer(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
        patch_size=4,
        hidden_size=512,
        num_layers=4,
        num_heads=6,
        mlp_dim=1024,
        drop_rate=0.1, 
        attn_drop_rate=0.0,
    )


def vit_b16_32x32(num_classes=10, input_channels=3, input_size=32):
    return VisionTransformer(
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
        patch_size=4,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        drop_rate=0.1,
        attn_drop_rate=0.0,
    )