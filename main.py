from const import *
from trainers import *
from utils import Data
import pickle
data = Data()


class CnnExp():
    def __init__(self) -> None:
        # self.datasets = BIG
        # self.data = Data()
        pass

    def debug(self, dataset=MNIST, tag=BAYES):
        trainer = Trainer(dataset)
        if tag == BAYES:
            with open(f"result/{dataset}_{tag}.json", "w") as f:
                hparams = json.load(f)
        trainer.reset_model(hparams)
        trainer.train(tag)
        return

    def cal_hparams(self, dataset=MNIST):
        print(dataset)
        trainer = Trainer(dataset)
        # print("bayes")
        # trainer.bayes()
        # print("zoopt")
        # trainer.zoopt()
        # print("rand")
        # trainer.rand()
        # print("ga")
        # trainer.ga()
        # print("pso")
        # trainer.pso()
        print("hb")
        trainer.hyper_band()
        return


if __name__ == "__main__":
    exp = CnnExp()
    exp.cal_hparams(MNIST)
    exp.cal_hparams(SVHN)
    exp.cal_hparams(CIFAR10)
    exp.cal_hparams(CIFAR100)
    pass
