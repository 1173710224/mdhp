from const import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.interpolate import make_interp_spline


class Parser():
    def __init__(self, model, dataset) -> None:
        self.dataset = dataset
        self.model = model
        path = self.get_default_path()
        with open(path, "rb") as f:
            self.data = pickle.load(f)
        pass

    def read(self):
        return self.data

    def reset(self, model, dataset):
        self.dataset = dataset
        self.model = model
        return

    def get_default_path(self):
        if self.dataset in BIG:
            son_dir = "big"
        elif self.dataset in SMALL:
            son_dir = "small"
        path = "result/{}/{}-{}".format(son_dir, self.model, self.dataset)
        return path


if __name__ == "__main__":
    # parser = Parser(SGD, IRIS)
    parser = Parser(ADAM, WINE)
    print(parser.data["loss"][:100])
    # for item in parser.data:
    #     print(type(item), len(item))
    #     print(parser.data[item])
    #     # print(item)
    #     break
    pass
