import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss, MSELoss, DataParallel
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
from train_triplet import train as triplet_train


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
        alp_lambda = 0.5
        train_set = CIFAR10(root="./data", train=True,
                            download=True, transform=transform_train)
        val_set = CIFAR10(root="./data", train=True,
                          download=True, transform=transform_val)
    else:
        alp_lambda = 0.5
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
    if opt.metric == "arcface":
        model.fc = AAML(
            model.fc.in_features, num_classes, device, s=opt.s, m=opt.m)
    elif opt.metric == "cosface":
        model.fc = LMCL(
            model.fc.in_features, num_classes, device, s=opt.s, m=opt.m)
    elif opt.metric == "sphereface":
        model.fc = AMSL(
            model.fc.in_features, num_classes, device, m=opt.m)
    else:
        model.fc = Softmax(model.fc.in_features, num_classes)

    model.to(device)
    model = DataParallel(model)

    # set optimizer
    criterion = CrossEntropyLoss()
    mse_criterion = MSELoss(reduction="sum")

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
    for epoch in range(opt.epoch):
        for phase in ["train", "val"]:
            total_examples, total_correct, total_loss = 0, 0, 0

            if phase == "train":
                model.train()
            else:
                model.eval()

            for ii, data in enumerate(data_loaders[phase]):
                # load data batch to device
                images, labels = data

                # perform adversarial attack update to images
                if opt.train_mode == "at" or opt.train_mode == "alp":
                    adv_images = pgd(
                        model, images, labels, 8. / 255, 2. / 255, 7, device)
                else:
                    pass
                
                # at train mode prediction
                if opt.train_mode == "at":
                    adv_images = adv_images.to(device)
                    labels = labels.to(device).long()
                    predictions = model(adv_images, labels)

                    # get loss
                    loss = criterion(predictions, labels)
                    optimizer.zero_grad()

                # alp train mode prediction
                elif opt.train_mode == "alp":
                    images = images.to(device)
                    adv_images = adv_images.to(device)
                    labels = labels.to(device).long()

                    # logits for clean and adversarial images
                    predictions = model(images, labels)
                    adv_predictions = model(adv_images, labels)

                    # get ce loss
                    ce_loss = criterion(adv_predictions, labels)

                    # get alp loss
                    alp_loss = mse_criterion(
                        adv_predictions, predictions) / adv_predictions.size(0)

                    # get overall loss
                    loss = ce_loss + alp_lambda * alp_loss
                    optimizer.zero_grad()

                # clean train mode prediction
                else:
                    images = images.to(device)
                    labels = labels.to(device).long()
                    predictions = model(images, labels)

                    # get loss
                    loss = criterion(predictions, labels)
                    optimizer.zero_grad()

                # only take step if in train phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # accumulate train or val results
                predictions = torch.argmax(predictions, 1)
                total_examples += predictions.size(0)
                total_correct += predictions.eq(labels).sum().item()
                total_loss += loss.item()

                # print accumulated train/val results at end of epoch
                if ii == len(data_loaders[phase]) - 1:
                    acc = total_correct / total_examples
                    loss = total_loss / len(data_loaders[phase])
                    print("{}: Epoch -- {} Loss -- {:.6f} Acc -- {:.6f} Lr -- {:.4f}".format(
                        phase, epoch, loss, acc, optimizer.param_groups[0]['lr']))

                    if phase == "train":
                        loss = total_loss / len(data_loaders[phase])
                        scheduler.step(loss)
                    else:
                        print("")

    # save model after training for opt.epoch
    if opt.test_bb:
        save_model(model, opt.dataset + "_bb", opt.train_mode,
                   opt.metric, opt.backbone)
    else:
        save_model(model, opt.dataset, opt.train_mode,
                   opt.metric, opt.backbone)


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()

    if opt.metric == "triplet":
        # perform training using arguments
        np.random.seed(42)
        torch.manual_seed(42)
        opt.test_bb = False
        triplet_train(opt)

        # perform training of second model to use for black box attack
        np.random.seed(24)
        torch.manual_seed(24)
        opt.test_bb = True
        triplet_train(opt)

    else:
        # perform training using arguments
        np.random.seed(42)
        torch.manual_seed(42)
        opt.test_bb = False
        train(opt)

        # perform training of second model to use for black box attack
        np.random.seed(24)
        torch.manual_seed(24)
        opt.test_bb = True
        train(opt)
