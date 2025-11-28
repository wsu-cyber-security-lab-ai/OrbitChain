from typing import Callable

import torch
from torch import nn
from torchvision import models

from .config import DEVICE
from .utils import set_seed



class MnistCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_mnist_model(num_classes: int = 10) -> nn.Module:
    return MnistCNN(num_classes=num_classes).to(DEVICE)


def build_resnet18(num_classes: int) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        m = models.resnet18(weights=weights)
    except Exception:
        m = models.resnet18(pretrained=True)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m.to(DEVICE)


def get_model_builder(dataset_name: str, num_classes: int) -> Callable[[], nn.Module]:
    """
    Returns a zero-arg function that builds a fresh model.
    """
    name = dataset_name.lower()
    if name == "mnist":
        return lambda: build_mnist_model(num_classes)
    if name in ("eurosat", "ucmerced"):
        return lambda: build_resnet18(num_classes)
    raise ValueError(f"Unknown dataset name for model builder: {dataset_name}")
