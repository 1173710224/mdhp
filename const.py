# datasets
CIFAR10 = "cifar-10-batches-py"
CIFAR100 = "cifar-100-python"
MNIST = "MNIST"
SVHN = "SVHN"
CAR = "car"
WINE = "wine"
IRIS = "iris"
AGARICUS = "agaricus_lepiota"
DATASETS = [CIFAR10, CIFAR100, MNIST, SVHN, WINE, CAR, IRIS, AGARICUS]
LARGE = [MNIST, CIFAR10, CIFAR100, SVHN]
SMALL = [WINE, CAR, IRIS, AGARICUS]
NUMIMAGE = {
    MNIST: 60000,
    SVHN: 73257,
    CIFAR10: 50000,
    CIFAR100: 50000
}
CIFAR10MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10STD = (0.2470, 0.2435, 0.2616)
CIFAR100MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MNISTMEAN = (0.1307,)
MNISTSTD = (0.3081,)
SVHNMEAN = (0.4376821219921112, 0.4437697231769562, 0.4728044271469116)
SVHNSTD = (0.19803012907505035, 0.2010156363248825, 0.19703614711761475)

# optimizers
ADAM = "adam"
DSA = "dsa"
SGD = "sgd"
MOMENTUM = "momentum"
RMSPROP = "rmsprop"
ADAMAX = "adamax"
ADAMW = "adamw"
ADAGRAD = "adagrad"
ADADELTA = "adadelta"
HD = "hypergradient"
OPTIMIZERS = [
    ADAM, ADAMW, ADAMAX,
    ADADELTA, ADAGRAD, SGD,
    RMSPROP, MOMENTUM, HD, DSA]
OPTIMIZERS2LABEL = {
    ADAM: "Adam",
    ADAMW: "AdamW",
    ADAMAX: "Adamax",
    ADADELTA: "AdaDelta",
    ADAGRAD: "AdaGrad",
    SGD: "SGD",
    RMSPROP: "RMSPprop",
    MOMENTUM: "Momentum",
    DSA: "DSA",
    HD: "HD"}

# metrics
ACCU = "accu"
RECALL = "recall"
PRECISION = "precision"
F1SCORE = "f1score"
TRAINLOSS = "trainloss"
VALLOSS = "valloss"
TRACK = "track"
LOSSOLDLR = "loss_last_lr"
LOSSNEWLR = "loss_tmp_lr"
CONFLICT = "conflict"
INITDICT = {ACCU: [],
            RECALL: [],
            PRECISION: [],
            F1SCORE: [],
            VALLOSS: [],
            TRAINLOSS: [],
            TRACK: [],
            LOSSOLDLR: [],
            LOSSNEWLR: [],
            CONFLICT: [],
            }

# models
FMP = "fmp"
DNN = "dnn"
MLP = "mlp"
RESNET = "resnet"
SUM = "sum"


# """hyper-params"""
A = 1
B = 1000
# B = 95
MINIBATCHEPOCHS = 200
EPOCHSDENSE = 1000
EPOCHSTEP1 = 15
EPOCHSTEP2 = 20
MAXEPOCHS = 1000
SUMEPOCH = 1000
TRACKEPOCH = 1000
EPSILON = 1e-20
SUMNUMS = [1000, 10000, 100000]
P_MOMENTUM = 0.9
BATCHSIZE = 64
NAME2BATCHSIZE = {
    CIFAR10: 128,
    CIFAR100: 32,
    MNIST: 128,
    SVHN: 64,
}

# mode
MINI = "mini"

if __name__ == "__main__":
    pass
