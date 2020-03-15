import os
import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import load_model


np.random.seed(42)
torch.manual_seed(42)


def test(opt):
    # set device to cpu/gpu
    if opt.use_gpu == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get CIFAR10/CIFAR100 test set
    if opt.dataset == "CIFAR10":
        test_set = CIFAR10(root="./data", train=False,
                           download=True, transform=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        test_set = CIFAR100(root="./data", train=False,
                            download=True, transform=transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    num_classes = np.unique(test_set.targets).shape[0]

    # get test dataloader
    test_loader = DataLoader(test_set,
                             batch_size=opt.batch_size,
                             num_workers=2)

    print("Test iteration batch size: {}".format(opt.batch_size))
    print("Test iterations per epoch: {}".format(len(test_loader)))

    model = load_model(opt.dataset, opt.metric, opt.backbone).eval()
    metric_fc = load_model(opt.dataset, opt.metric +
                           "_fc", opt.backbone).eval()

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    # get prediction results for model
    y_true, y_pred = [], []
    for ii, data in enumerate(test_loader):
        # load data batch to device
        data_input, label = data
        data_input = data_input.to(device)
        label = label.to(device).long()
        # get feature embedding from resnet
        feature = model(data_input)
        # get prediction
        output = metric_fc(feature, label)

        # accumulate test results
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        y_true.append(label)
        y_pred.append(output)

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    print(classification_report(y_true, y_pred))
    return y_true, y_pred


if __name__ == '__main__':
    # load in arguments defined in config/config.py
    opt = Config()

    # perform training using arguments
    y_true, y_pred = test(opt)
