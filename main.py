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

    def generate_sample_for_mehp(self, dataset):
        trainer = Trainer(dataset)
        trainer.generate_training_sample()
        return

    def train_mehp(self, dataset):
        trainer = Trainer(dataset)
        trainer.train_mapper()
        return

    def cal_params_for_mehp(self, dataset):
        trainer = Trainer(dataset)
        train_embedding = trainer.embedding_dataset(trainer.train_loader)
        mapper = Mapper()
        mapper.load_state_dict(torch.load(f'dehp/{dataset}_model'))
        hps = mapper(torch.Tensor(train_embedding).unsqueeze(0))
        hps = mapper.generate(hps)
        hps = hps.squeeze(0)
        with open(f"hparams/{dataset}_{MEHP}.json", "w") as f:
            json.dump(list(hps), f)
        return


if __name__ == "__main__":
    exp = CnnExp()
    # # calculate h-parameters of baselines
    # exp.cal_hparams(MNIST)
    # exp.cal_hparams(SVHN)
    # exp.cal_hparams(CIFAR10)
    # exp.cal_hparams(CIFAR100)
    exp.generate_sample_for_mehp(MNIST)
    pass
