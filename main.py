from const import *
from trainers import *
from models import MobileNetV2
from utils import Data
import pickle
data = Data()


class CnnExp():
    def __init__(self) -> None:
        # self.datasets = BIG
        # self.data = Data()
        pass

    def debug(self, dataset=MNIST, opt=ADAM):
        train_loader, test_loader, input_channel, ndim, nclass = self.data.get(
            dataset)
        trainer = Trainer()
        trainer.train()
        # trainer.save_metrics(f"result/big/fmp_{dataset}_{opt}")
        return


if __name__ == "__main__":
    exp = CnnExp()
    exp.debug()
    pass
