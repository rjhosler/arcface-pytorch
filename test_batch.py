import numpy as np
from sklearn.metrics import accuracy_score
import torch
from test import test
from config import Config


test_results = {}

opt = Config()

opt.metric = "softmax"  # softmax, contrastive, triplet, center, aaml
opt.train_mode = "clean"  # clean, at, alp
opt.dataset = "CIFAR100"  # CIFAR10, CIFAR100, tiny_imagenet
opt.backbone = "resnet18"  # resnet18, resnet34

for test_mode in ["clean", "fgsm", "bim", "cw", "pgd_7", "pgd_20", "20pgd20", "mim", "bb"]:
    np.random.seed(42)
    torch.manual_seed(42)

    if test_mode == "20pgd20":
        opt.test_mode = "pgd_20"
        opt.test_restarts = 20
        opt.test_bb = False

    elif test_mode == "bb":
        opt.test_mode = "pgd_7"
        opt.test_restarts = 1
        opt.test_bb = True

    else:
        opt.test_mode = test_mode
        opt.test_restarts = 1
        opt.test_bb = False

    # perform training using arguments
    y_true, y_pred = test(opt)

    test_results[test_mode] = accuracy_score(y_true, y_pred)

print("\nDataset -- {}, Metric -- {}, Train Mode -- {}, Backbone -- {}".format(
    opt.dataset, opt.metric, opt.train_mode, opt.backbone))
for mode, result in test_results.items():
    print("{} -- {}".format(mode, result))
