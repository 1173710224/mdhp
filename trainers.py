import json
from pickletools import optimize
from models import ResNet, Hyperband, AutoEncoder, Mapper
from utils import Data, num_image, Sampler, MehpDataset
from torch.utils.data import DataLoader
import torch
from const import *
import torch.nn.functional as F
import warnings
import time
from random import randint, random
from zoopt import Dimension, Objective, Parameter, Opt
from bayes_opt import BayesianOptimization
from sko.GA import GA
from sko.PSO import PSO
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import numpy as np
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, dataset=CIFAR10, hparams=[64, 128, 256, 512, 2, 2, 2, 2], epoch=MINIBATCHEPOCHS) -> None:
        self.epoch = epoch
        train_loader, test_loader, input_channel, ndim, nclass = Data().get(dataset)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.input_channel = input_channel
        self.ndim = ndim
        self.nclass = nclass
        self.num_image = num_image(train_loader)
        self.dataset = dataset
        self.model = ResNet(input_channel, ndim, nclass, hparams=hparams)
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        pass

    def train(self, tag=""):
        self.optimizier = torch.optim.SGD(
            self.model.parameters(), lr=0.05, momentum=P_MOMENTUM, weight_decay=0.0005)
        # self.lr_sch = torch.optim.lr_scheduler.MultiStepLR(self.optimizier,
        #    milestones=[self.epoch * 0.5, self.epoch * 0.75], gamma=0.1)
        # self.lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizier, T_max=200)
        # self.optimizier = torch.optim.Adam(
        #     self.model.parameters(), lr=0.001)
        self.model.train()
        losses = []
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
                print(loss.item())
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
            # self.lr_sch.step()
            avg_loss = loss_sum * 1.0/self.num_image
            losses.append(losses)
            print("Epoch~{}->train_loss:{},time:{}s".format(i+1,
                  round(avg_loss, 4), round(time.time() - start, 4)))
        with open(f"result/{self.dataset}_{tag}.json", "w") as f:
            json.dump({ACCU: self.val(), TRAINLOSS: losses}, f)
        return loss_sum

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
            self.model.eval()
            preds = self.model(imgs)
            ncorrect += torch.sum(preds.max(1)[1].eq(label).double())
            nsample += len(label)
        return ncorrect/nsample

    def reset_model(self, hparams):
        self.model = ResNet(self.input_channel, self.ndim,
                            self.nclass, hparams=hparams)
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = "cuda"
        else:
            self.model.cpu()
            self.device = "cpu"
        return

    def objective(self, iter=10):
        self.optimizier = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=0.0001)
        self.model.train()
        ret = 0
        for i in range(iter):
            loss_sum = 0
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
            ret = loss_sum * 1.0/self.num_image
            if (i + 1) % 5 == 0:
                print(f"Epoch~{i + 1}->loss:{ret}.")
        return ret

    def bayes(self):
        def max_obj(c1, c2, c3, c4, b1, b2, b3, b4):
            hparams = [int(c1), int(c2), int(c3), int(
                c4), int(b1), int(b2), int(b3), int(b4)]
            self.reset_model(hparams)
            return -self.objective()
        self.loss_set = []
        optimizer = BayesianOptimization(max_obj, {
            C1: (32, 64+0.99),
            C2: (64, 128+0.99),
            C3: (128, 256+0.99),
            C4: (256, 512+0.99),
            B1: (2, 3+0.99),
            B2: (2, 4+0.99),
            B3: (2, 6+0.99),
            B4: (2, 3+0.99),
        })
        st = time.perf_counter()
        optimizer.maximize(init_points=5, n_iter=ITERATIONS-5)
        with open(f"hparams/{self.dataset}_{BAYES}.json", "w") as f:
            json.dump([int(optimizer.max['params'][tmp])
                      for tmp in HPORDER], f)
        print(f"time: {time.perf_counter() - st}")
        return time.perf_counter() - st, optimizer.max["params"], -optimizer.max["target"]

    def zoopt(self):
        def min_obj(solution):
            hparams = solution.get_x()
            self.reset_model(hparams)
            return self.objective()
        st = time.perf_counter()
        solution = Opt.min(Objective(min_obj, Dimension(
            8,
            [[32, 64],
             [64, 128],
             [128, 256],
             [256, 512],
             [2, 3],
             [2, 4],
             [2, 6],
             [2, 3]],
            [False, False, False,
             False, False, False,
             False, False]
        )), Parameter(budget=ITERATIONS))
        with open(f"hparams/{self.dataset}_{ZOOPT}.json", "w") as f:
            json.dump(solution.get_x(), f)
        print(f"time: {time.perf_counter() - st}")
        return time.perf_counter() - st, solution.get_x(), solution.get_value()

    def rand(self):
        best_hparams = None
        best_loss = INF
        st = time.perf_counter()
        for _ in range():
            hparams = [
                randint(32, 64),
                randint(64, 128),
                randint(128, 256),
                randint(256, 512),
                randint(2, 3),
                randint(2, 4),
                randint(2, 6),
                randint(2, 3),
            ]
            self.reset_model(hparams)
            loss = self.objective()
            if loss < best_loss:
                best_loss = loss
                best_hparams = hparams
        with open(f"hparams/{self.dataset}_{RAND}.json", "w") as f:
            json.dump(best_hparams, f)
        print(f"time: {time.perf_counter() - st}")
        return time.perf_counter() - st, best_hparams, best_loss

    def ga(self):
        def schaffer(param_values):
            hparams = [int(tmp) for tmp in param_values]
            self.reset_model(hparams)
            return self.objective()
        ga = GA(func=schaffer, n_dim=8, size_pop=10, max_iter=10, prob_mut=0.001,
                lb=[32, 64, 128, 256, 2, 2, 2, 2],
                ub=[64+0.99, 128+0.99, 256+0.99, 512+0.99, 3+0.99, 4+0.99, 6+0.99, 3+0.99], precision=1e-7)
        st = time.perf_counter()
        best_params, best_loss = ga.run()
        with open(f"hparams/{self.dataset}_{GENETICA}.json", "w") as f:
            json.dump([int(tmp) for tmp in best_params], f)
        print(f"time: {time.perf_counter() - st}")
        return time.perf_counter() - st, best_params, best_loss

    def pso(self):
        def schaffer(param_values):
            hparams = [int(tmp) for tmp in param_values]
            self.reset_model(hparams)
            return self.objective()
        ga = PSO(func=schaffer, n_dim=8, size_pop=10, max_iter=10, prob_mut=0.001,
                 lb=[32, 64, 128, 256, 2, 2, 2, 2],
                 ub=[64+0.99, 128+0.99, 256+0.99, 512+0.99, 3+0.99, 4+0.99, 6+0.99, 3+0.99], precision=1e-7)
        st = time.perf_counter()
        best_params, best_loss = ga.run()
        with open(f"hparams/{self.dataset}_{PARTICLESO}.json", "w") as f:
            json.dump([int(tmp) for tmp in best_params], f)
        print(f"time: {time.perf_counter() - st}")
        return time.perf_counter() - st, best_params, best_loss

    def hyper_band(self):
        def get_params_conv():
            hparams = sample({
                C1: hp.choice(C1, [i for i in range(32, 64+1)]),
                C2: hp.choice(C2, [i for i in range(64, 128+1)]),
                C3: hp.choice(C3, [i for i in range(128, 256+1)]),
                C4: hp.choice(C4, [i for i in range(256, 512+1)]),
                B1: hp.choice(B1, [i for i in range(2, 3+1)]),
                B2: hp.choice(B2, [i for i in range(2, 4+1)]),
                B3: hp.choice(B3, [i for i in range(2, 6+1)]),
                B4: hp.choice(B4, [i for i in range(2, 3+1)]),
            })
            return [int(hparams[tmp]) for tmp in HPORDER]

        def try_params_conv(hparams, iter):
            self.reset_model(hparams)
            loss = self.objective()
            result = {}
            result['loss'] = loss
            return result
        hb = Hyperband(get_params_conv, try_params_conv, it_n=9)
        st = time.perf_counter()
        results = hb.run()
        best_loss = results["best_loss"]
        best_hparams = results["best_hparams"]
        with open(f"hparams/{self.dataset}_{HYPERBAND}.json", "w") as f:
            json.dump(best_hparams, f)
        print(f"time: {time.perf_counter() - st}")
        return time.perf_counter() - st, best_hparams, best_loss

    def embedding_dataset(self, dataloader):
        encoder = AutoEncoder(self.input_channel, self.ndim)
        if torch.cuda.is_available():
            encoder.cuda()
        optimizer = torch.optim.Adam(encoder.parameters(), weight_decay=1e-5)
        encoder.train()
        for _ in range(EMBEDDINGEPOCH):
            for data in dataloader:
                img, _ = data
                if torch.cuda.is_available():
                    img.cuda()
                output = encoder(img)
                loss = F.mse_loss(output, img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        embeddings = []
        for data in dataloader:
            img, _ = data
            if torch.cuda.is_available():
                img.cuda()
            embeddings.append(encoder.embedding(img))
        embedding = torch.cat(embeddings, dim=0).mean(dim=0)
        return embedding.detach().cpu().numpy()

    def optimal_hparams(self, dataloader):
        def objective(iter=INFEPOCH):
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=0.0001)
            self.model.train()
            ret = 0
            for i in range(iter):
                loss_sum = 0
                for imgs, label in dataloader:
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        label = label.cuda()
                    preds = self.model(imgs)
                    loss = F.cross_entropy(preds, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(imgs)
                ret = loss_sum
                if (i + 1) % 5 == 0:
                    print(f"Epoch~{i + 1}->loss:{ret}.")
            return ret

        def max_obj(c1, c2, c3, c4, b1, b2, b3, b4):
            hparams = [int(c1), int(c2), int(c3), int(
                c4), int(b1), int(b2), int(b3), int(b4)]
            self.reset_model(hparams)
            return -objective()
        self.loss_set = []
        optimizer = BayesianOptimization(max_obj, {
            C1: (32, 64+0.99),
            C2: (64, 128+0.99),
            C3: (128, 256+0.99),
            C4: (256, 512+0.99),
            B1: (2, 3+0.99),
            B2: (2, 4+0.99),
            B3: (2, 6+0.99),
            B4: (2, 3+0.99),
        })
        st = time.perf_counter()
        optimizer.maximize(init_points=5, n_iter=INFITER)
        print(f"time: {time.perf_counter() - st}")
        return [int(optimizer.max['params'][tmp]) for tmp in HPORDER]

    def generate_training_sample(self, num_sample=1000, dataset_size=500):
        x = []
        y = []
        sampler = Sampler(self.dataset)
        for i in num_sample:
            loader, _, _, _ = sampler.fetch(dataset_size)
            embedding = self.embedding_dataset(loader)
            hparams = self.optimal_hparams(loader)
            x.append(embedding)
            y.append(hparams)
        torch.save((x, y), f"mehp/{self.dataset}")
        return

    def train_mapper(self,):
        train_dataset = MehpDataset(self.dataset)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=128, shuffle=True,)
        mapper = Mapper()
        mapper.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=0.0001)
        for epoch in range(1000):
            sum_loss = 0
            num_embedding = 0
            for embedding, hps in train_loader:
                if torch.cuda.is_available():
                    embedding.cuda()
                    hps.cuda()
                preds = mapper(embedding)
                optimizer.zero_grad()
                loss0 = 0
                for i in range(8):
                    exec(
                        f'loss{i+1} = loss{i} + F.cross_entropy(preds[{i}], hps[:,{i}])/8')
                exec(f'loss{8}.backward()')
                optimizer.step()
                sum_loss += eval(f'loss{8}') * len(embedding)
                num_embedding += len(embedding)
            sum_loss /= num_embedding
            print(f"Epoch~{epoch}->loss:{sum_loss}")
        torch.save(mapper.state_dict(), f"mehp/{self.dataset}_model")
        return


if __name__ == "__main__":
    # trainer = Trainer(MNIST)
    # sampler = Sampler(trainer.dataset)
    # loader, _, _, _ = sampler.fetch()
    # embedding = trainer.embedding_dataset(loader)
    # torch.save(([embedding, embedding], [[64, 128, 256, 512, 2, 2, 2, 2], [64, 128, 256, 512, 2, 2, 2, 2]]),
    #            f"mehp/{trainer.dataset}")
    train_dataset = MehpDataset(MNIST)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=True,)
    mapper = Mapper()
    mapper.train()
    for embedding, hps in train_loader:
        if torch.cuda.is_available():
            embedding.cuda()
            hps.cuda()
        preds = mapper(embedding)
        loss0 = 0
        for i in range(8):
            exec(
                f'loss{i+1} = loss{i} + F.cross_entropy(preds[{i}], hps[:,{i}])/8')
        exec(f'loss{8}.backward()')
        print(eval(f'loss{8}').item())
    pass
