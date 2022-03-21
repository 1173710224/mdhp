from turtle import forward
from typing import List
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, Flatten, Linear, PReLU, FractionalMaxPool2d, RNN
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from const import *
from utils import Data
import torch
from random import random
from math import log, ceil
import numpy as np


class Hyperband:

    def __init__(self, get_params_function, try_params_function, it_n):
        self.get_params = get_params_function
        self.try_params = try_params_function
        self.max_iter = it_n  	# maximum iterations per configuration
        self.eta = 3			# defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = {}
        self.results["best_loss"] = INF
        self.results["best_hparams"] = 0
        self.best_loss = INF
        self.counter = 0
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):
        # num = 0
        print(reversed(range(self.s_max + 1)))
        for s in reversed(range(self.s_max + 1)):
            print(s)

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            # n=30
            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = n * self.eta ** (-i)
                n_iterations = int(r * self.eta ** (i))

                losses = []
                for t in T:
                    self.counter += 1
                    if dry_run:
                        result = {'loss': random()}
                    else:
                        result = self.try_params(t, n_iterations)
                    if result['loss'] >= INF:
                        continue

                    loss = result['loss']
                    losses.append(loss)
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter
                        self.results["best_loss"] = loss
                        self.results["best_hparams"] = t
                indices = np.argsort(losses)
                T = [T[i] for i in indices]
                T = T[0:int(n_configs / self.eta)]
        return self.results


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, ndim, num_classes, block=BasicBlock, num_blocks=[2, 2, 2, 2], hparams=[64, 128, 256, 512, 2, 2, 2, 2]):
        super(ResNet, self).__init__()
        self.in_planes = hparams[0]
        self.conv1 = nn.Conv2d(input_channel, hparams[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hparams[0])
        self.layer1 = self._make_layer(
            block, hparams[0], hparams[4], stride=1)
        self.layer2 = self._make_layer(
            block, hparams[1], hparams[5], stride=2)
        self.layer3 = self._make_layer(
            block, hparams[2], hparams[6], stride=2)
        self.layer4 = self._make_layer(
            block, hparams[3], hparams[7], stride=2)
        self.linear = nn.Linear(hparams[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def reset_parameters(self):
        return


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
        x = self.decoder(x)
        return x

    def embedding(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return x


class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.scale = nn.Sequential(
            Linear(32, 64),
            nn.Tanh()
        )
        self.rnn = RNN(input_size=64, hidden_size=128, num_layers=1)
        self.trans1 = nn.Sequential(
            Linear(128, 64),
            Linear(64, 32+1),
        )
        self.trans2 = nn.Sequential(
            Linear(128, 64+1),
        )
        self.trans3 = nn.Sequential(
            Linear(128, 128+1),
        )
        self.trans4 = nn.Sequential(
            Linear(128, 256+1),
        )
        self.trans5 = nn.Sequential(
            Linear(128, 64),
            Linear(64, 32),
            Linear(32, 2),
        )
        self.trans6 = nn.Sequential(
            Linear(128, 64),
            Linear(64, 32),
            Linear(32, 3),
        )
        self.trans7 = nn.Sequential(
            Linear(128, 64),
            Linear(64, 32),
            Linear(32, 5),
        )
        self.trans8 = nn.Sequential(
            Linear(128, 64),
            Linear(64, 32),
            Linear(32, 2),
        )
        return

    def forward(self, x: torch.Tensor):
        x = self.scale(x)
        x = x.unsqueeze(0).repeat(8, 1, 1)
        outputs, _ = self.rnn(x)
        hps = []
        for i, output in enumerate(outputs):
            hps.append(eval(f"self.trans{i+1}")(output))
        return hps

    def generate(self, hps):
        res = []
        res.append(torch.argmax(hps[0], 1).unsqueeze(1) + 32*1)
        res.append(torch.argmax(hps[1], 1).unsqueeze(1) + 32*2)
        res.append(torch.argmax(hps[2], 1).unsqueeze(1) + 32*4)
        res.append(torch.argmax(hps[3], 1).unsqueeze(1) + 32*8)
        res.append(torch.argmax(hps[4], 1).unsqueeze(1) + 2)
        res.append(torch.argmax(hps[5], 1).unsqueeze(1) + 2)
        res.append(torch.argmax(hps[6], 1).unsqueeze(1) + 2)
        res.append(torch.argmax(hps[7], 1).unsqueeze(1) + 2)

        return torch.cat(res, dim=1).detach().cpu().numpy()


if __name__ == "__main__":
    # data = Data()
    # # loader, _, input_channel, _, _ = data.load_svhn()
    # loader, _, input_channel, ndim, _ = data.load_mnist()
    # # loader, _, input_channel, _, _ = data.load_cifar10()
    # model = AutoEncoder(input_channel, ndim)
    # for imgs, label in loader:
    #     output = model(imgs)
    #     # print(output.size(), imgs.size())
    #     # loss = F.mse_loss(output, imgs)
    #     print(model.embedding(imgs).size())
    #     break

    # rnn = nn.RNN(input_size=2, hidden_size=3, num_layers=2,
    #              bias=False, nonlinearity='relu')
    # input = torch.ones(1, 1, 2)
    # h0 = torch.zeros(2, 1, 3)
    # output, hn = rnn(input, h0)
    # print("x", input.reshape(1, 2))
    # print("output", output.reshape((1, 3)))
    # print("output hidden", hn)
    # for name, param in rnn.named_parameters():
    #     print(name, param)

    x = torch.rand(10, 32)
    mapper = Mapper()
    hps = mapper(x)
    for hp in hps:
        print(hp.size())
    # print(hps[4:])
    hps = mapper.generate(hps)
    print(hps)
    pass
