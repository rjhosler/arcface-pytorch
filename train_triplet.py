import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import save_model, load_model
from models.metrics import Softmax, AAML, LMCL, AMSL
from models.attacks import fgsm, bim, pgd, mim, cw
from models.resnet_cifar import resnet18, resnet34
from models.triplet_loss import batch_all_triplet_loss


np.random.seed(42)
torch.manual_seed(42)


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
        transforms.ToTensor(),
        transforms.RandomErasing(),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    ])

    # get CIFAR10/CIFAR100 train/val set
    if opt.dataset == "CIFAR10":
        margin = 0.03
        lambda_loss = [2, 0.001]
        train_set = CIFAR10(root="./data", train=True,
                            download=True, transform=transform_train)
        val_set = CIFAR10(root="./data", train=True,
                          download=True, transform=transform_val)
    else:
        margin = 0.03
        lambda_loss = [2, 0.001]
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

    # get backbone model
    if opt.backbone == "resnet18":
        model = resnet18(pretrained=False)
    else:
        model = resnet34(pretrained=False)

    # set metric loss function
    model.fc = Softmax(model.fc.in_features, num_classes)

    model.to(device)

    # set optimizer
    criterion = CrossEntropyLoss()

    # set LR scheduler
    if opt.scheduler == "decay":
        if opt.optimizer == "sgd":
            optimizer = SGD([{"params": model.parameters()}],
                            lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
        else:
            optimizer = Adam([{"params": model.parameters()}],
                             lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)

    else:
        if opt.optimizer == "sgd":
            optimizer = SGD([{"params": model.parameters()}],
                            lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
        else:
            optimizer = Adam([{"params": model.parameters()}],
                             lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10)

    # train/val loop
    for epoch in range(opt.max_epoch):
        for phase in ["train", "val"]:
            acc_accum = []
            loss_accum = []

            if phase == "train":
                model.train()
            else:
                model.eval()

            for ii, data in enumerate(data_loaders[phase]):
                # load data batch to device
                images, labels = data

                # get feature embedding from resnet and prediction
                images = images.to(device)
                labels = labels.to(device).long()
                features = model.feature(images)

                # get triplet loss (margin 0.03 CIFAR10/100, 0.01 TinyImageNet from paper)
                tpl_loss, _, mask = batch_all_triplet_loss(
                    labels, features, margin)

                # get feature norm loss
                norm = features.mm(features.t()).diag()
                norm_loss = norm[mask.nonzero()[0]] + \
                    norm[mask.nonzero()[1]] + norm[mask.nonzero()[2]]
                norm_loss = torch.sum(norm_loss) / mask.nonzero()[0].shape[0]

                # get cross-entropy loss (only considering anchor examples)
                anchor_images = images[np.unique(mask.nonzero()[0])]
                anchor_labels = labels[np.unique(mask.nonzero()[0])]
                predictions = model(anchor_images, anchor_labels)
                ce_loss = criterion(predictions, anchor_labels)

                # combine cross-entropy loss, triplet loss and feature norm lossusing lambda weights
                loss = ce_loss + lambda_loss[0] * \
                    tpl_loss + lambda_loss[1] * norm_loss
                optimizer.zero_grad()

                # only take step if in train phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # accumulate train or val results
                predictions = predictions.data.cpu().numpy()
                predictions = np.argmax(predictions, axis=1)
                anchor_labels = anchor_labels.data.cpu().numpy()
                acc_accum.append(predictions == anchor_labels)
                loss_accum.append(loss.item())

                # print accumulated train/val results at end of epoch
                if ii == len(data_loaders[phase]) - 1:
                    acc = np.sum(np.concatenate(
                        acc_accum).astype(int)) / np.concatenate(acc_accum).astype(int).shape[0]
                    print("{}: Epoch -- {} Loss -- {:.6f} Acc -- {:.6f} Lr -- {:.4f}".format(
                        phase, epoch, np.average(loss_accum), acc, optimizer.param_groups[0]['lr']))

                    if phase == "train":
                        loss_total = np.mean(loss_accum)
                        scheduler.step(loss_total)
                    else:
                        print("")

    # save model after training for opt.epoch
    if opt.test_bb:
        save_model(model, opt.dataset + "_bb", opt.train_mode,
                   opt.metric, opt.backbone)
    else:
        save_model(model, opt.dataset, opt.train_mode,
                   opt.metric, opt.backbone)
