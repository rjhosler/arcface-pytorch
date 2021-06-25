import numpy as np
from sklearn.metrics import accuracy_score
import torch
from test import test
from config import Config


test_results = {}

opt = Config()

#opt.metric = "center"  # softmax, contrastive, triplet, center, aaml
TRAIN_MODES = ["clean"]  # clean, at, alp
opt.dataset = "CIFAR10"  # CIFAR10, CIFAR100, tiny_imagenet
opt.backbone = "resnet18"  # resnet18, resnet34
BB_METRICS = ["softmax", "contrastive", "triplet", "center", "aaml"]
METRICS = ["softmax"]

opt.test_mode = "fgsm"
opt.test_bb = True
opt.test_restarts = 20

opt.m = 0.2
opt.s = 2

# bb variations
for train_metric in METRICS:
    for train_mode in TRAIN_MODES:
        for bb_metric in BB_METRICS:
            np.random.seed(42)
            torch.manual_seed(42)
            
            opt.train_mode = train_mode
            opt.bb_metric = bb_metric
            opt.metric = train_metric

            y_true, y_pred = test(opt)

            test_results[bb_metric+" attacks "+opt.metric+"_"+opt.train_mode] = accuracy_score(y_true, y_pred)

#print("\nDataset -- {}, Metric -- {}, Train Mode -- {}, Backbone -- {}".format(opt.dataset, opt.metric, opt.train_mode, opt.backbone))
for mode, result in test_results.items():
    print("{} -- {}".format(mode, result))
