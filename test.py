import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.nn import DataParallel, CrossEntropyLoss
from torch.nn.functional import normalize
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import load_model
from models.attacks import fgsm, bim, pgd


np.random.seed(42)
torch.manual_seed(42)


def test(opt):
    # set device to cpu/gpu
    if opt.use_gpu == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # get CIFAR10/CIFAR100 test set
    if opt.dataset == "CIFAR10":
        test_set = CIFAR10(root="./data", train=False,
                           download=True, transform=transform_test)
    else:
        test_set = CIFAR100(root="./data", train=False,
                            download=True, transform=transform_test)
    num_classes = np.unique(test_set.targets).shape[0]

    # get test dataloader
    test_loader = DataLoader(test_set,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers,
                             shuffle=False)

    print("Test iteration batch size: {}".format(opt.batch_size))
    print("Test iterations per epoch: {}".format(len(test_loader)))

    model = load_model(opt.dataset, opt.train_mode, opt.metric, opt.backbone)
    metric_fc = load_model(opt.dataset, opt.train_mode,
                           opt.metric + "_fc", opt.backbone)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    # get prediction results for model
    y_true, y_pred = [], []
    acc_accum = []
    for ii, data in enumerate(test_loader):
        # load data batch to device
        data_input, label = data

        # perform adversarial attack update to images
        if opt.test_mode == "fgsm":
            data_input = fgsm(
                model, metric_fc, data_input, label, 8. / 255., device)
        elif opt.test_mode == "bim":
            data_input = bim(
                model, metric_fc, data_input, label, 8. / 255., 2 / 255., 7, device)
        elif opt.test_mode == "pgd_7":
            data_input = pgd(
                model, metric_fc, data_input, label, 8. / 255., 2 / 255., 7, device)
        elif opt.test_mode == "pgd_20":
            data_input = pgd(
                model, metric_fc, data_input, label, 8. / 255., 2 / 255., 20, device)
        else:
            pass

        # normalize input images
        data_input = data_input.to(device)
        label = label.to(device).long()
        for i in range(data_input.shape[0]):
            data_input[i] = transforms.functional.normalize(
                data_input[i], [0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010])

        # get feature embedding from resnet and prediction
        feature = model(data_input)
        output = metric_fc(feature, label)

        # accumulate test results
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        acc_accum.append(output == label)
        y_true.append(label)
        y_pred.append(output)

    acc = np.sum(np.concatenate(
        acc_accum).astype(int)) / np.concatenate(acc_accum).astype(int).shape[0]

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    print(classification_report(y_true, y_pred))
    print("Accuracy: {}".format(acc))
    return y_true, y_pred


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()

    # perform training using arguments
    y_true, y_pred = test(opt)
