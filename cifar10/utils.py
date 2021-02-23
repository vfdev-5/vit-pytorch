import os

import torch
import torch.optim as optim
from torchvision import models
from torchvision import datasets
import torchvision.transforms as T

from vit import VisionTransformer, vit_b16


mean_vals = (0.485, 0.456, 0.406)
std_vals = (0.229, 0.224, 0.225)


cifar10_train_transform = T.Compose(
    [
        T.Pad(4),
        T.RandomCrop(32, fill=128),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean_vals, std_vals),
    ]
)

cifar10_test_transform = T.Compose([T.ToTensor(), T.Normalize(mean_vals, std_vals),])


def create_train_transform(size, rand_aug, with_erasing=False):

    tfs = [
        T.RandomResizedCrop(size, interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
    ]

    if rand_aug is None:
        tfs.append(T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0))
    else:
        pass
        
    tfs += [
        T.ToTensor(),
        T.Normalize(mean_vals, std_vals),        
    ]

    if with_erasing:
        tfs.append(T.RandomErasing(value="random"))

    return T.Compose(tfs)


def create_test_transform(size):
    return T.Compose([
        T.Resize(int((256 / 224) * size), interpolation=3),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean_vals, std_vals),   
    ])


def get_train_test_datasets(path, rescale_size=None, rand_aug=None, with_erasing=False):
    if not os.path.exists(path):
        os.makedirs(path)
        download = True
    else:
        download = True if len(os.listdir(path)) < 1 else False

    if rescale_size is None:
        assert rand_aug is None
        train_transform = cifar10_train_transform
        test_transform = cifar10_test_transform
    else:
        train_transform = create_train_transform(rescale_size, rand_aug, with_erasing)
        test_transform = create_test_transform(rescale_size)

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
    elif name in ["vit_b16", ]:
        fn = __dict__[name]
    else:
        raise RuntimeError(f"Unknown model name {name}")

    model = fn(num_classes=10)
    if torch_scripted:
        model = torch.jit.script(model)

    return model


def get_optimizer(name, model, learning_rate=None, weight_decay=None):
    opt_configs = {}
    if name == "adam":
        fn = optim.Adam
    elif name == "sgd":
        fn = optim.SGD
        opt_configs["nesterov"] = True
    elif name == "adamw":
        fn = optim.AdamW
    else:
        raise RuntimeError(f"Unknown optmizer name {name}")

    if learning_rate is not None:
        opt_configs["lr"] = learning_rate
    if weight_decay is not None:
        opt_configs["weight_decay"] = weight_decay

    optimizer = fn(model.parameters(), **opt_configs)
    return optimizer


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
