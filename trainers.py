from models import MobileNetV2
from utils import Data, num_image
import torch
from const import *
import torch.nn.functional as F
import warnings
import time
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, dataset=CIFAR10, h_params=[], epoch=MINIBATCHEPOCHS) -> None:
        self.epoch = epoch
        train_loader, test_loader, input_channel, ndim, nclass = Data().get(dataset)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_image = num_image(train_loader)
        self.data = Data()
        self.h_params = h_params
        self.model = MobileNetV2()
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self):
        self.optimizier = torch.optim.SGD(
            self.model.parameters(), momentum=0.9)
        self.lr_sch = torch.optim.lr_scheduler.MultiStepLR(self.optimizier,
                                                           milestones=[self.epoch * 0.5, self.epoch * 0.75], gamma=0.1)
        self.model.train()
        for i in range(self.epoch):
            loss_sum = 0
            start = time.time()
            for imgs, label in self.train_loader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    label = label.cuda()
                else:
                    imgs = imgs.cpu()
                    label = label.cpu()
                preds = self.model(imgs)
                loss = F.cross_entropy(preds, label)
                self.optimizier.zero_grad()
                loss.backward()
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
            self.lr_sch.step()
            avg_loss = loss_sum * 1.0/self.num_image
            print("Epoch~{}->train_loss:{},val_accu:{},time:{}s".format(i+1, round(avg_loss, 4),
                  self.val(), round(time.time() - start, 4)))
        return

    def val(self):
        ncorrect = 0
        nsample = 0
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            else:
                imgs = imgs.cpu()
                label = label.cpu()
            # print(imgs.device, label.device)
            self.model.eval()
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
        return ncorrect/nsample

    def objective(self):
        self.multi_loss_seq = []
        accu_sum = 0
        for i in range(VALTIMES):
            self.model.reset_parameters()
            self.train()
            self.multi_loss_seq.append(self.loss_sequence)
            accu_sum += float(self.val())
        return accu_sum/VALTIMES


if __name__ == "__main__":

    pass
