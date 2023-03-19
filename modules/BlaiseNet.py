import torch
import torch.nn as nn
from loader import CustomDataset
from torch.utils.data import random_split
import torchvision.transforms as transforms

IMAGE_PATH = "../data/dhs_train"
CSV_PATH = "../data/train.csv"
dataset = CustomDataset(path=IMAGE_PATH, target=CSV_PATH, transform=transforms.ToTensor())

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])


class BlaiseNet(nn.Module):
    def __init__(self, in_channels: int = 8) -> None:
        super(BlaiseNet, self).__init__()
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding="same")
        pool1 = nn.MaxPool2d(2)
        conv2 = nn.Conv2d(in_channels=8, out_channels=12, stride=2, kernel_size=3)
        pool2 = nn.MaxPool2d(2)
        conv3 = nn.Conv2d(in_channels=12, out_channels=16, stride=3, kernel_size=3)
        pool2 = nn.MaxPool2d(2)
        conv4 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=3)
        pool2 = nn.MaxPool2d(2)
        conv5 = nn.Conv2d(in_channels=20, out_channels=16, kernel_size=5, padding="same")
        pool2 = nn.MaxPool2d(2)
        conv6 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding='same')
        pool2 = nn.MaxPool2d(2)
        conv7 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, padding='same')
        pool2 = nn.MaxPool2d(2)
        conv8 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding='same')

        self.linear = nn.Sequential(nn.LazyLinear(out_features=10),
                                    nn.BatchNorm1d(10),
                                    nn.ReLU(inplace=True),
                                    nn.LazyLinear(out_features=10),
                                    nn.BatchNorm1d(10),
                                    nn.ReLU(inplace=True),
                                    nn.LazyLinear(out_features=1))

        self.convs = nn.Sequential(
            conv1,nn.MaxPool2d(2), conv2, conv3, conv4, conv5, conv6, conv7, conv8
        )

    def forward(self, X):
        X = self.convs(X)
        X = torch.flatten(X, 1)
        X = self.linear(X)
        return X


if __name__ == "__main__":
    net = BlaiseNet()
    random_image = torch.randn(2, 8, 255, 255)
    num = net.forward(random_image)
    print(num)
    print(net)
