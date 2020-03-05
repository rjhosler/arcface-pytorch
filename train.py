import os
import numpy as np
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from models import *
from config import Config


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + "_" + str(iter_cnt) + ".pth")
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == "__main__":
    opt = Config()
    if opt.use_gpu == True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transforms.Compose(
                                                [transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    print("{} train iters per epoch:".format(len(trainloader)))

    criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == "resnet18":
        model = resnet18()
    elif opt.backbone == "resnet34":
        model = resnet34()
    elif opt.backbone == "resnet50":
        model = resnet50()

    if opt.metric == "add_margin":
        metric_fc = AddMarginProduct(512, opt.num_classes, device, s=30, m=0.35)
    elif opt.metric == "arc_margin":
        metric_fc = ArcMarginProduct(512, opt.num_classes, device, s=30, m=0.5)
    elif opt.metric == "sphere":
        metric_fc = SphereProduct(512, opt.num_classes, device, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == "sgd":
        optimizer = torch.optim.SGD([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    for i in range(opt.max_epoch):
        if i > 0:
            scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                print("train epoch {} iter {} loss {} acc {}".format(
                    i, ii, loss.item(), acc))

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
