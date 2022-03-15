from torch.utils.data import dataset
from utils import Data
from const import *
import numpy as np


class Encoder():
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.encode = None
        self.data_loader = Data()
        pass

    def get_encode(self):
        return self.encode


class StatisticEncoder(Encoder):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        return

    def generate_encode(self):
        '''
        计算每个类别x每个特征的（最大值、最小值、中位数、平均数）
        '''
        encode = []
        classified_samples = {}
        if self.dataset in SMALL:
            train_data, _, _, _ = self.data_loader.get(self.dataset)
            x_train, y_train = train_data
            x_train = x_train.cpu().numpy()
            y_train = y_train.cpu().numpy()
            for index in range(len(y_train)):
                if not classified_samples.__contains__(y_train[index]):
                    classified_samples[y_train[index]] = [x_train[index]]
                else:
                    classified_samples[y_train[index]].append(x_train[index])
            print(classified_samples.keys())  # , classified_samples.values())
            for iter_class in classified_samples.keys():
                samples = classified_samples[iter_class]
                samples = np.array(samples)
                if len(samples) == 0:
                    continue
                for index in range(len(samples[0])):
                    encode.append(np.max(samples[:, index]))
                    encode.append(np.min(samples[:, index]))
                    encode.append(np.mean(samples[:, index]))
                    encode.append(np.median(samples[:, index]))
            self.encode = encode
        elif self.dataset in BIG:
            pass
        return

    def aggregate_samples(self):
        classified_samples = {}
        for index in range():
            pass
        return


if __name__ == "__main__":
    encoder = StatisticEncoder(WINE)
    encoder.generate_encode()
    print(encoder.encode)
    pass
