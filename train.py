import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss, DataParallel, Dropout
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import save_model, load_model, resnet_fc
from models.metrics import Softmax, AAML, LMCL, AMSL
from models.resnet_cifar10 import resnet18, resnet34, resnet50


np.random.seed(42)
torch.manual_seed(42)


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):
    #https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee

    # Scaler: we can adapt this if we do not want the triangular CLR
    def scaler(x): return 1.

    # Lambda function to calculate the LR
    def lr_lambda(it): return min_lr + (max_lr -
                                        min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = np.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def train(opt):
    # set device to cpu/gpu
    if opt.use_gpu == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data transformations for data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10, scale=(1, 1.5), shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
        transforms.RandomErasing(),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010]),
    ])

    # get CIFAR10/CIFAR100 train/val set
    if opt.dataset == "CIFAR10":
        train_set = CIFAR10(root="./data", train=True,
                            download=True, transform=transform_train)
        val_set = CIFAR10(root="./data", train=True,
                          download=True, transform=transform_val)
    else:
        train_set = CIFAR100(root="./data", train=True,
                             download=True, transform=transform_train)
        val_set = CIFAR100(root="./data", train=True,
                           download=True, transform=transform_val)
    num_classes = np.unique(train_set.targets).shape[0]

    # set stratified train/val split
    idx = list(range(len(train_set.targets)))
    train_idx, val_idx, _, _ = train_test_split(
        idx, train_set.targets, test_size=opt.val_split, random_state=42)

    # get train/val samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # get train/val dataloaders
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=opt.batch_size,
                              num_workers=opt.num_workers)
    val_loader = DataLoader(val_set,
                            sampler=val_sampler,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers)

    data_loaders = {"train": train_loader, "val": val_loader}

    print("Train iteration batch size: {}".format(opt.batch_size))
    print("Train iterations per epoch: {}".format(len(train_loader)))

    # get backbone model, set embedding size (if 512, take raw feature from backbone model)
    if opt.backbone == "resnet18":
        model = resnet18(pretrained=False)
        model.fc = resnet_fc(model.fc.in_features, opt.emb_feat_size)
    elif opt.backbone == "resnet34":
        model = resnet34(pretrained=False)
        model.fc = resnet_fc(model.fc.in_features, opt.emb_feat_size)
    elif opt.backbone == "resnet50":
        model = resnet50(pretrained=False)
        model.fc = resnet_fc(model.fc.in_features, opt.emb_feat_size)

    # set metric loss function
    if opt.metric == "arcface":
        metric_fc = AAML(
            opt.emb_feat_size, num_classes, device, s=opt.s, m=opt.m)
    elif opt.metric == "cosface":
        metric_fc = LMCL(
            opt.emb_feat_size, num_classes, device, s=opt.s, m=opt.m)
    elif opt.metric == "sphereface":
        metric_fc = AMSL(
            opt.emb_feat_size, num_classes, device, m=opt.m)
    else:
        metric_fc = Softmax(opt.emb_feat_size, num_classes)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    # set optimizer
    criterion = CrossEntropyLoss()

    # set LR scheduler
    if opt.scheduler == "decay":
        if opt.optimizer == "sgd":
            optimizer = SGD([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                            lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
        else:
            optimizer = Adam([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                             lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
    else:
        if opt.optimizer == "sgd":
            optimizer = SGD([{"params": model.parameters()}, {
                            "params": metric_fc.parameters()}], lr=1.)
        else:
            optimizer = Adam([{"params": model.parameters()}, {
                             "params": metric_fc.parameters()}], lr=1.)
        step_size = 4 * len(train_loader)
        clr = cyclical_lr(step_size, min_lr=opt.cycle_lr /
                          opt.cycle_factor, max_lr=opt.cycle_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr, clr])

    # train/val loop
    best_val_acc = -np.inf
    best_epoch = 0
    for epoch in range(opt.max_epoch):
        for phase in ["train", "val"]:
            acc_accum = []
            loss_accum = []

            if phase == "train":
                model.train()
                metric_fc.train()
            else:
                model.eval()
                metric_fc.eval()

            for ii, data in enumerate(data_loaders[phase]):
                # load data batch to device
                data_input, label = data
                data_input = data_input.to(device)
                label = label.to(device).long()

                # get feature embedding from resnet
                feature = model(data_input)

                # get prediction and loss
                output = metric_fc(feature, label)
                loss = criterion(output, label)
                optimizer.zero_grad()

                # only take step if in train phase
                if phase == "train":
                    loss.backward()
                    scheduler.step()
                    optimizer.step()

                # accumulate train or val results
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc_accum.append(output == label)
                loss_accum.append(loss.item())

                # print accumulated train/val results at end of epoch
                if ii == len(data_loaders[phase]) - 1:
                    acc = np.sum(np.concatenate(
                        acc_accum).astype(int)) / np.concatenate(acc_accum).astype(int).shape[0]

                    print("{}: Epoch -- {} Loss -- {:.6f} Acc -- {:.6f} Lr -- {:.4f}".format(
                        phase, epoch, np.average(loss_accum), acc, scheduler.get_last_lr()[0]))

                    # check earlystopping convergence critera
                    if phase == "val":
                        # save model to checkpoints dir if training improved val loss, update curr_patience
                        if acc > best_val_acc:
                            print("val accuracy improved: {:.6f} to {:.6f} ({:.6f})\n".format(
                                best_val_acc, acc, acc - best_val_acc))

                            curr_patience = 0
                            best_val_acc = acc
                            best_epoch = epoch
                            save_model(model, opt.dataset,
                                       opt.metric, opt.backbone)
                            save_model(metric_fc, opt.dataset,
                                       opt.metric + "_fc", opt.backbone)

                        else:
                            print("val accuracy not improved: {:.6f} to {:.6f}, (+{:.6f})\n".format(
                                best_val_acc, acc, best_val_acc - acc))

                            curr_patience += 1

                        # terminate model if earlystopping patience exceeded
                        if curr_patience > opt.patience:
                            print("converged after {} epochs, loading best model from epoch {}".format(
                                epoch, best_epoch))

                            return (load_model(opt.dataset, opt.metric, opt.backbone),
                                    load_model(opt.dataset, opt.metric + "_fc", opt.backbone))


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()

    # perform training using arguments
    train(opt)
