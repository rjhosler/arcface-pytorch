import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.nn import DataParallel, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import load_model
from torch.optim import lr_scheduler, SGD, Adam

os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.random.seed(42)
torch.manual_seed(42)


def fgsm_attack(model, metric_fc, images, labels, eps, device):
    loss = CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    features = model(images)
    outputs = metric_fc(features, labels)

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images


def test(opt):
    # set device to cpu/gpu
    if opt.use_gpu == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
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

    model = load_model(opt.dataset, opt.metric, opt.backbone)
    metric_fc = load_model(opt.dataset, opt.metric +
                           "_fc", opt.backbone)

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
        if opt.mode == "fgsm":
            data_input = fgsm_attack(
                model, metric_fc, data_input, label, 8./255., device)

        # load images and labels to device
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
