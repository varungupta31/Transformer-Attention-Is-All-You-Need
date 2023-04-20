import numpy as np
import torch
from deit import DataEfficientImageTransformer as DEIT
from vit import VisionTransformer
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms, datasets
import sys

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(224),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(30),
#     #Normalization values picked up from a discussion @ https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
#     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
# ])

# trainset = datasets.CIFAR100(root = './dataset/train', train = True, download=True, transform=transform)

# valid_ratio = 0.04
# n_train_samples = int(len(trainset) * (1-valid_ratio))
# n_valid_samples = len(trainset) - n_train_samples

# print(f"There are {n_train_samples} Train samples, and {n_valid_samples} in the Dataset.")

# sys.exit()

custom_config = {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

model_custom = DEIT(**custom_config)

print(summary(model_custom, (3, 224, 224)))

