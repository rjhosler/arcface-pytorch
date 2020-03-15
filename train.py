import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import save_model, load_model, resnet_fc
from models.metrics import Softmax, AAML, LMCL, AMSL


np.random.seed(42)
torch.manual_seed(42)


def train(opt):
    # set device to cpu/gpu
    if opt.use_gpu == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # get CIFAR10/CIFAR100 train set
    if opt.dataset == "CIFAR10":
        train_set = CIFAR10(root="./data", train=True,
                            download=True, transform=transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        train_set = CIFAR100(root="./data", train=True,
                             download=True, transform=transforms.Compose(
                                 [transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
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
                              batch_size=opt.batch_size,
                              num_workers=opt.num_workers,
                              sampler=train_sampler)
    val_loader = DataLoader(train_set,
                            batch_size=opt.batch_size,
                            num_workers=2,
                            sampler=val_sampler)

    data_loaders = {"train": train_loader, "val": val_loader}
    data_lengths = {"train": len(train_idx), "val": len(val_idx)}

    print("Train iteration batch size: {}".format(opt.batch_size))
    print("Train iterations per epoch: {}".format(len(train_loader)))

    # get backbone model, set embedding size (if 512, take raw feature from backbone model)
    if opt.backbone == "resnet18":
        model = resnet18(pretrained=True)
        model.fc = resnet_fc(opt.emb_feat_size)
    elif opt.backbone == "resnet34":
        model = resnet34(pretrained=True)
        model.fc = resnet_fc(opt.emb_feat_size)
    elif opt.backbone == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = resnet_fc(opt.emb_feat_size)

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
    if opt.optimizer == "sgd":
        optimizer = SGD([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                        lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = Adam([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                         lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=opt.lr_step, gamma=0.1)

    # train/val loop
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(opt.max_epoch):
        if epoch > 0:
            scheduler.step()
        for phase in ["train", "val"]:
            acc_accum = []
            loss_accum = []

            if phase == "train":
                model.train(True)
            else:
                model.train(False)

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
                    optimizer.step()

                # accumulate train or val results
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc_accum.append(output == label)
                loss_accum.append(loss.item())

                # only print accumulated train results based on print_freq argument
                if phase == "train" and ii % opt.print_freq == 0:
                    acc = np.sum(np.concatenate(
                        acc_accum).astype(int)) / np.concatenate(acc_accum).astype(int).shape[0]
                    print("{}: Epoch -- {} Iter -- {} Loss -- {:.6f} Acc -- {:.6f}".format(
                        phase, epoch, ii, np.average(loss_accum), acc))

                # print accumulated train/val results at end of epoch
                if ii == len(data_loaders[phase]) - 1:
                    acc = np.sum(np.concatenate(
                        acc_accum).astype(int)) / np.concatenate(acc_accum).astype(int).shape[0]
                    print("{}: Epoch -- {}  Loss -- {:.6f} Acc -- {:.6f}".format(
                        phase, epoch, np.average(loss_accum), acc))

                    # check earlystopping convergence critera
                    if phase == "val":
                        # save model to checkpoints dir if training improved val loss, update curr_patience
                        if np.average(loss_accum) < best_val_loss:
                            print("val loss improved: {:.6f} to {:.6f} ({:.6f})\n".format(
                                best_val_loss, np.average(loss_accum), np.average(loss_accum) - best_val_loss))
                            curr_patience = 0
                            best_val_loss = np.average(loss_accum)
                            best_epoch = epoch
                            save_model(model, opt.dataset,
                                       opt.metric, opt.backbone)
                            save_model(metric_fc, opt.dataset,
                                       opt.metric + "_fc", opt.backbone)

                        else:
                            print("val loss not improved: {:.6f} to {:.6f}, (+{:.6f})\n".format(
                                best_val_loss, np.average(loss_accum), np.average(loss_accum) - best_val_loss))
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
    model, metric_fc = train(opt)
