from zmq import device
from const import *
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import pickle


class Data():
    def __init__(self) -> None:
        self.datasets = DATASETS
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        return

    def load_cifar10(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10MEAN, CIFAR10STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10MEAN, CIFAR10STD),
        ])
        data_root_path = "data/"
        train_dataset = datasets.CIFAR10(root=data_root_path, train=True,
                                         transform=train_transform,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=data_root_path, train=False,
                                        transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[CIFAR10], shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader, 3, 32, 10

    def load_cifar100(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100MEAN, CIFAR100STD),
        ])
        data_root_path = "data/"
        train_dataset = datasets.CIFAR100(root=data_root_path, train=True,
                                          transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(root=data_root_path, train=False,
                                         transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[CIFAR100], shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader, 3, 32, 100

    def load_mnist(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MNISTMEAN, MNISTSTD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNISTMEAN, MNISTSTD),
        ])
        data_root_path = "data/"
        train_dataset = datasets.MNIST(root=data_root_path, train=True,
                                       transform=train_transform, download=True)
        test_dataset = datasets.MNIST(root=data_root_path, train=False,
                                      transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[MNIST], shuffle=True,
                                  num_workers=4,
                                  )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True,
                                 )
        return train_loader, test_loader, 1, 28, 10

    def load_svhn(self):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHNMEAN, SVHNSTD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHNMEAN, SVHNSTD),
        ])
        data_root_path = "data/SVHN/"
        train_dataset = datasets.SVHN(root=data_root_path, split="train",
                                      transform=train_transform, download=True)
        test_dataset = datasets.SVHN(root=data_root_path, split="test",
                                     transform=test_transform, download=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=NAME2BATCHSIZE[SVHN], shuffle=True,
                                  num_workers=4)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCHSIZE, shuffle=True)
        return train_loader, test_loader, 3, 32, 10

    def load_wine(self):
        LabelIndex = 0
        path = "data/wine/wine.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, 1:],
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        counter = {0: 0, 1: 0, 2: 0}
        train_index = []
        test_index = []
        for index in range(len(dataset)):
            item = dataset[index]
            if counter[int(item[-1])] < 40:
                counter[int(item[-1])] += 1
                train_index.append(index)
            else:
                test_index.append(index)
        train_data = dataset.index_select(
            0, torch.tensor(train_index, device=self.device))
        test_data = dataset.index_select(
            0, torch.tensor(test_index, device=self.device))
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        return (x_train, y_train), (x_test, y_test), 13, 3

    def load_car(self):
        LabelIndex = 6
        path = "data/car/car.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, :-1]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        # counter = {0: 0, 1: 0, 2: 0, 3: 0}
        # train_index = []
        # test_index = []
        # for index in range(len(dataset)):
        #     item = dataset[index]
        #     if counter[int(item[-1])] < 50:
        #         counter[int(item[-1])] += 1
        #         train_index.append(index)
        #     else:
        #         test_index.append(index)
        # train_data = dataset.index_select(0, torch.tensor(train_index))
        # test_data = dataset.index_select(0, torch.tensor(test_index))
        # x_train = train_data[:, :-1]
        # y_train = train_data[:, -1]
        # x_test = test_data[:, :-1]
        # y_test = test_data[:, -1]
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test), 21, 4

    def load_iris(self):
        LabelIndex = 4
        path = "data/iris/iris.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((df.values[:, :-1],
                                  sp.LabelEncoder().fit_transform(
            df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test), 4, 3

    def load_agaricus_lepiota(self):
        LabelIndex = 0
        path = "data/agaricus-lepiota/agaricus-lepiota.data"
        df = pd.read_csv(path, header=None)
        dataset = np.column_stack((sp.OneHotEncoder(sparse=False).fit_transform(df.values[:, 1:11]),
                                   sp.OneHotEncoder(sparse=False).fit_transform(
                                       df.values[:, 12:]),
                                   sp.LabelEncoder().fit_transform(df[[LabelIndex]].values)))
        dataset = np.array(dataset, dtype=float)
        dataset = torch.Tensor(dataset)
        if torch.cuda.is_available():
            dataset = dataset.cuda()
        x_train, x_test, y_train, y_test = train_test_split(
            dataset[:, :-1], dataset[:, -1:].reshape(len(dataset)), test_size=0.2, random_state=0)
        return (x_train, y_train), (x_test, y_test), 112, 2

    def get(self, dataset):
        if dataset == MNIST:
            return self.load_mnist()
        if dataset == SVHN:
            return self.load_svhn()
        if dataset == CIFAR10:
            return self.load_cifar10()
        if dataset == CIFAR100:
            return self.load_cifar100()
        if dataset == IRIS:
            return self.load_iris()
        if dataset == WINE:
            return self.load_wine()
        if dataset == CAR:
            return self.load_car()
        if dataset == AGARICUS:
            return self.load_agaricus_lepiota()
        return None


def num_image(loader):
    res = 0
    for _, label in loader:
        res += len(label)
    return res


def get_opt(opt, model, dataset=None):
    if opt == ADAM:
        return torch.optim.Adam(
            model.parameters(), lr=0.001)
    if opt == DSA:
        return DiffSelfAdaptDotplus(model.parameters(), lr_init=-4.6, meta_lr=0.3)
        # return DiffSelfAdaptDagger(model.parameters(), lr_init=-4.6, meta_lr=0.1)
        # return DiffSelfAdapt(model.parameters(), lr_init=-4.6, meta_lr=0.01)
        # return MomentumDiffSelfAdapt(model.parameters(), lr_init=-6.9, meta_lr=0.1, momentum=0.2)
    if opt == HD:
        if dataset in SMALL:
            return HypergraDient(model.parameters())
        return HypergraDient(model.parameters(), lr_init=0.001, meta_lr=1e-4)
    if opt == ADAMW:
        return torch.optim.AdamW(model.parameters(), lr=0.001)

    if opt == ADAMAX:
        return torch.optim.Adamax(model.parameters(), lr=0.001)
    if opt == ADAGRAD:
        return torch.optim.Adagrad(model.parameters(), lr=0.001)
    if opt == ADADELTA:
        return torch.optim.Adadelta(model.parameters())

    if opt == SGD:
        return torch.optim.SGD(model.parameters(), lr=0.001)
    if opt == RMSPROP:
        return torch.optim.RMSprop(model.parameters(), lr=0.001)
    if opt == MOMENTUM:
        return torch.optim.SGD(model.parameters(), lr=0.001, momentum=P_MOMENTUM, weight_decay=0.0001)
    return None


def get_res(dataset, opt):
    if dataset in LARGE:
        with open(f"result/big/fmp_{dataset}_{opt}") as f:
            return pickle.load(f)
    elif dataset in SMALL:
        with open(f"result/small/mlp_{dataset}_{opt}") as f:
            return pickle.load(f)
    return None


def get_scheduler(opt, optimizer):
    if opt in [SGD, MOMENTUM]:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[MINIBATCHEPOCHS * 0.5, MINIBATCHEPOCHS * 0.75], gamma=0.1)
    return None


if __name__ == "__main__":
    pass
