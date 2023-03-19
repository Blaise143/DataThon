import torch
import torch.nn as nn
from loader import CustomDataset
from torch.utils.data import random_split
import torchvision.transforms as transforms

IMAGE_PATH = "../data/dhs_train"
CSV_PATH = "../data/train.csv"
dataset = CustomDataset(path=IMAGE_PATH,target=CSV_PATH, transform=transforms.ToTensor())

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset,[train_size, test_size])


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv(X)


class BlaiseNet(nn.Module):
    def __init__(self, in_channels: int = 8) -> None:
        super(BlaiseNet, self).__init__()
        conv1 = ConvBlock(in_channels=in_channels, out_channels=8, stride=2)
        conv2 = ConvBlock(in_channels=8, out_channels=12, stride=3)
        conv3 = ConvBlock(in_channels=12, out_channels=16, stride=5)
        conv4 = ConvBlock(in_channels=16, out_channels=20, stride=5)
        conv5 = ConvBlock(in_channels=20, out_channels=16, stride=5)
        conv6 = ConvBlock(in_channels=16, out_channels=12, stride=3)
        conv7 = ConvBlock(in_channels=12, out_channels=8, stride=3)
        conv8 = ConvBlock(in_channels=8, out_channels=4, stride=3)

        self.linear = nn.Sequential(nn.LazyLinear(out_features=10),
                               nn.BatchNorm2d(10),
                               nn.ReLU(inplace=True),
                               nn.LazyLinear(out_features=10),
                               nn.BatchNorm2d(10),
                               nn.ReLU(inplace=True),
                               nn.LazyLinear(out_features=1))

        self.convs = nn.Sequential(
            conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8
        )

    def forward(self, X):
        X = self.convs(X)
        X = torch.flatten(X, 1)
        X = self.linear(X)
        return X


if __name__ == "__main__":
    net = BlaiseNet()
    random_image = torch.randn(32,8,255,288)
    net.forward(random_image)
    print(net)
