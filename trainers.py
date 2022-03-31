import json
from models import ResNet, Hyperband, AutoEncoder, Mapper
# from DEHB.dehb import DEHB
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
from sklearn.metrics import precision_recall_fscore_support as metrics
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
            self.model.parameters(), lr=0.1, momentum=P_MOMENTUM, weight_decay=0.0001)
        self.lr_sch = torch.optim.lr_scheduler.MultiStepLR(self.optimizier,
                                                           milestones=[self.epoch * 0.5, self.epoch * 0.75], gamma=0.1)
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
                self.optimizier.step()
                loss_sum += loss.item() * len(imgs)
            self.lr_sch.step()
            avg_loss = loss_sum * 1.0/self.num_image
            losses.append(avg_loss)
            print("Epoch~{}->train_loss:{},time:{}s".format(i+1,
                  round(avg_loss, 4), round(time.time() - start, 4)))
        accu, f1, recall, precision = self.val()
        with open(f"result/{self.dataset}_{tag}.json", "w") as f:
            json.dump({ACCU: accu, F1SCORE: f1, RECALL: recall,
                      PRECISION: precision, TRAINLOSS: losses}, f)
        return loss_sum

    def val(self):
        self.model.eval()
        ncorrect = 0
        nsample = 0
        preds = []
        Y = []
        for imgs, label in self.test_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()
            tmp_pred = self.model(imgs)
            tmp = tmp_pred.detach().cpu().numpy()
            preds.extend([np.argmax(tmp[i]) for i in range(len(tmp))])
            Y.extend(label.detach().cpu().numpy())
            ncorrect += torch.sum(tmp_pred.max(1)[1].eq(label).double())
            nsample += len(label)
        p, r, f1, _ = metrics(preds, Y)
        return float((ncorrect/nsample).cpu()), list(f1), list(r), list(p)

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

    def objective(self, iter=40):
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
        for _ in range(ITERATIONS):
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
        ga = GA(func=schaffer, n_dim=8, size_pop=EAPOP, max_iter=EAITER, prob_mut=0.001,
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
        pso = PSO(func=schaffer, n_dim=8, pop=EAPOP, max_iter=EAITER,
                  lb=[32, 64, 128, 256, 2, 2, 2, 2],
                  ub=[64+0.99, 128+0.99, 256+0.99, 512+0.99, 3+0.99, 4+0.99, 6+0.99, 3+0.99], w=0.8, c1=0.5, c2=0.5)
        st = time.perf_counter()
        best_params, best_loss = pso.run()
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
            loss = self.objective(iter)
            result = {}
            result['loss'] = loss
            return result
        hb = Hyperband(get_params_conv, try_params_conv, it_n=27)
        st = time.perf_counter()
        results = hb.run()
        best_loss = results["best_loss"]
        best_hparams = results["best_hparams"]
        with open(f"hparams/{self.dataset}_{HYPERBAND}.json", "w") as f:
            json.dump(best_hparams, f)
        print(f"time: {time.perf_counter() - st}")
        return time.perf_counter() - st, best_hparams, best_loss

    # def dehb(self):
    #     def transform_space(param_space, configuration):
    #         assert len(configuration) == len(param_space)
    #         res = []
    #         for i, (k, v) in enumerate(param_space.items()):
    #             value = configuration[i]
    #             lower, upper = v[0], v[1]
    #             is_log = v[3]
    #             if is_log:
    #                 # performs linear scaling in the log-space
    #                 log_range = np.log(upper) - np.log(lower)
    #                 value = np.exp(np.log(lower) + log_range * value)
    #             else:
    #                 # linear scaling within the range of the parameter
    #                 value = lower + (upper - lower) * value
    #             if v[2] == int:
    #                 value = int(value)
    #             res.append(value)
    #         return res

    #     def target_function(config, budget, **kwargs):
    #         max_budget = kwargs["max_budget"]
    #         if budget is None:
    #             budget = max_budget
    #         config = transform_space(param_space, config)
    #         self.reset_model(config)
    #         st = time.perf_counter()
    #         loss = self.objective()
    #         cost = time.perf_counter() - st
    #         result = {
    #             "fitness": loss,  # DE/DEHB minimizes
    #             "cost": cost,
    #             "info": {
    #                 "test_score": loss,
    #                 "budget": budget
    #             }
    #         }
    #         return result

    #     param_space = {
    #         C1: [32, 64, int, False],
    #         C2: [64, 128, int, False],
    #         C3: [128, 256, int, False],
    #         C4: [256, 512, int, False],
    #         B1: [2, 3, int, False],
    #         B2: [2, 4, int, False],
    #         B3: [2, 6, int, False],
    #         B4: [2, 3, int, False],
    #     }
    #     # Declaring the fidelity range
    #     min_budget, max_budget = 2, 18
    #     st = time.perf_counter()
    #     dehb = DEHB(
    #         f=target_function,
    #         dimensions=len(param_space),
    #         min_budget=min_budget,
    #         max_budget=max_budget,
    #         n_workers=1,
    #         output_path="./dehb_out"
    #     )
    #     trajectory, runtime, history = dehb.run(
    #         fevals=ITERATIONS,
    #         verbose=False,
    #         save_intermediate=False,
    #         max_budget=dehb.max_budget,
    #         param_space=param_space
    #     )
    #     with open(f"hparams/{self.dataset}_{DEHBCONST}.json", "w") as f:
    #         json.dump(transform_space(param_space, dehb.inc_config), f)
    #     print(f"time: {time.perf_counter() - st}")
    #     return time.perf_counter()-st, transform_space(param_space, dehb.inc_config), dehb.inc_score

    def embedding_dataset(self, dataloader):
        encoder = AutoEncoder(self.input_channel, self.ndim)
        if torch.cuda.is_available():
            encoder.cuda()
        optimizer = torch.optim.Adam(encoder.parameters(), weight_decay=1e-5)
        encoder.train()
        for i in range(EMBEDDINGEPOCH):
            for data in dataloader:
                img, _ = data
                if torch.cuda.is_available():
                    img = img.cuda()
                output = encoder(img)
                loss = F.mse_loss(output, img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        embeddings = []
        for data in dataloader:
            img, _ = data
            if torch.cuda.is_available():
                img = img.cuda()
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
                if i % 24 == 0:
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
        }, verbose=1)
        st = time.perf_counter()
        optimizer.maximize(init_points=5, n_iter=INFITER-5)
        print(f"time: {time.perf_counter() - st}")
        return [int(optimizer.max['params'][tmp]) for tmp in HPORDER]

    def generate_training_sample(self, num_sample=1000, dataset_size=500):
        x = []
        y = []
        sampler = Sampler(self.dataset)
        for i in range(num_sample):
            loader, _, _, _ = sampler.fetch(dataset_size)
            embedding = self.embedding_dataset(loader)
            hparams = self.optimal_hparams(loader)
            x.append(embedding)
            y.append(hparams)
            print(f'{i}th->embedding:{embedding},hparams:{hparams}')
        torch.save((x, y), f"mehp/{self.dataset}_data")
        return

    def train_mapper(self,):
        train_dataset = MehpDataset(self.dataset)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=128, shuffle=True,)
        mapper = Mapper()
        mapper.train()
        # optimizer = torch.optim.SGD(mapper.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(mapper.parameters(), weight_decay=0.0001)
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
