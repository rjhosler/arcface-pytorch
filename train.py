import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import save_model, load_model
from models.metrics import Softmax, AAML, LMCL, AMSL
from models.attacks import fgsm, bim, pgd, mim
from models.resnet_cifar10 import resnet18, resnet34

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    # set LR scheduler
    if opt.scheduler == "decay":
        if opt.optimizer == "sgd":
            optimizer = SGD([{"params": model.parameters()}],
                            lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
        else:
            optimizer = Adam([{"params": model.parameters()}],
                             lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
        
    else:
        if opt.optimizer == "sgd":
            optimizer = SGD([{"params": model.parameters()}],
                            lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
        else:
            optimizer = Adam([{"params": model.parameters()}],
                             lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    # train/val loop
    best_val_acc = -np.inf
    best_epoch = 0
    for epoch in range(opt.max_epoch):
        if epoch > 0:
            scheduler.step(loss_total)
    
        for phase in ["train", "val"]:
            acc_accum = []
            loss_accum = []

            if phase == "train":
                model.train()
            else:
                model.eval()

            for ii, data in enumerate(data_loaders[phase]):
                # load data batch to device
                data_input, label = data

                # perform adversarial attack update to images
                if opt.test_mode == "fgsm":
                    data_input = fgsm(
                        model, data_input, label, 8. / 255, device)
                elif opt.test_mode == "bim":
                    data_input = bim(
                        model, data_input, label, 8. / 255, 2. / 255, 7, device)
                elif opt.test_mode == "pgd_7":
                    data_input = pgd(
                        model, data_input, label, 8. / 255, 2. / 255, 7, device)
                elif opt.test_mode == "pgd_20":
                    data_input = pgd(
                        model, data_input, label, 8. / 255, 2. / 255, 20, device)
                elif opt.test_mode == "mim":
                    data_input = mim(
                        model, data_input, label, 8. / 255, 2. / 255, 0.9, 40, device)
                else:
                    pass

                # normalize input images
                data_input = data_input.to(device)
                label = label.to(device).long()

                # get feature embedding from resnet and prediction
                output = model(data_input, label)

                # get loss
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

                # print accumulated train/val results at end of epoch
                if ii == len(data_loaders[phase]) - 1:
                    acc = np.sum(np.concatenate(
                        acc_accum).astype(int)) / np.concatenate(acc_accum).astype(int).shape[0]

                    print("{}: Epoch -- {} Loss -- {:.6f} Acc -- {:.6f} Lr -- {:.4f}".format(
                        phase, epoch, np.average(loss_accum), acc, optimizer.param_groups[0]['lr']))
                        
                    if phase == "train":
                        loss_total = np.mean(loss_accum)

                    # check earlystopping convergence critera
                    if phase == "val":
                        # save model to checkpoints dir if training improved val acc, update curr_patience
                        if acc > best_val_acc:
                            print("val accuracy improved: {:.6f} to {:.6f} ({:.6f})\n".format(
                                best_val_acc, acc, acc - best_val_acc))

                            curr_patience = 0
                            best_val_acc = acc
                            best_epoch = epoch
                            save_model(model, opt.dataset, opt.train_mode,
                                       opt.metric, opt.backbone)

                        else:
                            print("val accuracy not improved: {:.6f} to {:.6f}, (+{:.6f})\n".format(
                                best_val_acc, acc, best_val_acc - acc))

                            curr_patience += 1

                        # terminate model if earlystopping patience exceeded
                        if curr_patience > opt.patience:
                            print("converged after {} epochs, loading best model from epoch {}".format(
                                epoch, best_epoch))

                            return best_val_acc


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()

    # perform training using arguments
    best_val_acc = train(opt)
