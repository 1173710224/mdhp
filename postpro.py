from const import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import make_interp_spline
import json


class Parser():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        pass

    def read_data(self, model):
        with open(f'result/{self.dataset}_{model}.json', 'r') as f:
            data = json.load(f)
        return data

    def get_loss(self, model=BAYES):
        data = self.read_data(model)
        return data[TRAINLOSS]

    def get_metrics(self, model=BAYES):
        data = self.read_data(model)
        return model + " {} & {}$\sim${} & {}$\sim${} & {}$\sim${} \\\\".format(
            '%.2f' % (data[ACCU] * 100),
            '%.2f' % (min(data[F1SCORE]) *
                      100), '%.2f' % (max(data[F1SCORE]) * 100),
            '%.2f' % (min(data[RECALL]) *
                      100), '%.2f' % (max(data[RECALL]) * 100),
            '%.2f' % (min(data[PRECISION]) * 100), '%.2f' % (max(data[PRECISION]) * 100))


if __name__ == "__main__":
    models = ['resnet18', BAYES, ZOOPT, RAND, GENETICA, PARTICLESO, HYPERBAND,
              # MEHP
              ]
    parser = Parser(CIFAR100)
    for model in models:
        print(parser.get_metrics(model))
    pass
