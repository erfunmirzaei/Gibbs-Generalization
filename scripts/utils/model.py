import lightning.pytorch as pl
import torch
import torch.nn.functional as f
from torch import nn
from torchvision import models


class NNModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self._input_dim)
        x = self.l1(x)
        x = f.relu(x)
        x = self.l2(x)
        x = f.relu(x)
        x = self.l3(x)
        x = f.log_softmax(x, dim=1)
        return x


class ConvNNModel(nn.Module):
    def __init__(self, in_channels: int = 1, dataset="mnist"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        if dataset == "mnist":
            self.fc1 = nn.Linear(9216, 128)
        elif dataset == "cifar10":
            self.fc1 = nn.Linear(12544, 128)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.log_softmax(x, dim=1)
        return x


class ConvNN15Model(nn.Module):
    def __init__(self, in_channels: int = 1, dataset="mnist"):
        super().__init__()
        if dataset != "cifar10":
            raise ValueError(f"Unknown dataset: {dataset}")

        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = f.relu(x)
        x = self.conv4(x)
        x = f.relu(x)
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv5(x)
        x = f.relu(x)
        x = self.conv6(x)
        x = f.relu(x)
        x = self.conv7(x)
        x = f.relu(x)
        x = self.conv8(x)
        x = f.relu(x)
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv9(x)
        x = f.relu(x)
        x = self.conv10(x)
        x = f.relu(x)
        x = self.conv11(x)
        x = f.relu(x)
        x = self.conv12(x)
        x = f.relu(x)
        x = f.max_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.fc3(x)
        x = f.log_softmax(x, dim=1)
        return x


class GoogLeNet(pl.LightningModule):
    def __init__(self, num_classes=10, num_channels: int = 1):
        super().__init__()
        self.model = models.googlenet(weights="IMAGENET1K_V1", transform_input=False)
        if num_channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif num_channels != 3:
            raise ValueError(f"Invalid number of channels: {num_channels}")
        self.model.fc = nn.Linear(1024, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.log_softmax(x)
        return x


class ResNet(pl.LightningModule):
    def __init__(self, num_classes: int = 10, num_channels: int = 1):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        if num_channels == 1:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif num_channels != 3:
            raise ValueError(f"Invalid number of channels: {num_channels}")
        self.model.fc = nn.Linear(512, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.log_softmax(x)
        return x
