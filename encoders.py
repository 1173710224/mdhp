import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from utils import Data
import torch.nn.functional as F

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])

# dataset = MNIST('./data', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self, input_channel, ndim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 16, 3, stride=3,
                      padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3,
                               padding=1-int(ndim/32)),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, input_channel, 2, stride=2,
                               padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.encoder(x)
        print(x.size())
        x = self.decoder(x)
        return x

    def embedding(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return x


if __name__ == "__main__":
    data = Data()
    # loader, _, input_channel, _, _ = data.load_svhn()
    loader, _, input_channel, ndim, _ = data.load_mnist()
    # loader, _, input_channel, _, _ = data.load_cifar10()
    model = AutoEncoder(input_channel, ndim)
    for imgs, label in loader:
        output = model(imgs)
        # print(output.size(), imgs.size())
        # loss = F.mse_loss(output, imgs)
        print(model.embedding(imgs).size())
        break
    pass
