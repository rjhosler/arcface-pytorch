import os
import numpy as np
import torch
from torch.nn import Module, CrossEntropyLoss, Linear, DataParallel
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18, resnet34, resnet50
from models.metrics import AddMarginProduct, ArcMarginProduct, SphereProduct
from config.config import Config
from torch.utils.data.sampler import SubsetRandomSampler


class ResNetFC(Module):
    def __init__(self, in_features, out_features):
        super(ResNetFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        if self.in_features == self.out_features:
            return x
        else:
            return Linear(in_features=self.in_features, out_features=self.out_features)


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

    if opt.dataset == "CIFAR10":
        trainset = CIFAR10(root="./data", train=True,
                        download=True, transform=transforms.Compose(
                            [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        trainset = CIFAR100(root="./data", train=True,
                           download=True, transform=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    num_classes = np.unique(trainset.targets).shape[0]
    
    validation_split = 0.1
    dataset_len = len(trainset)
    indices = list(range(dataset_len))

    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    
    trainloader = DataLoader(trainset,
                             batch_size=opt.train_batch_size,
                             num_workers=opt.num_workers,
                             sampler=train_sampler)
                             
    validationloader = DataLoader(trainset,
                             batch_size=opt.train_batch_size, 
                             num_workers=opt.num_workers,
                             sampler=validation_sampler)
                             
    data_loaders = {"train": trainloader, "val": validationloader}
    data_lengths = {"train": len(train_idx), "val": val_len}

    print("Train iteration batch size: {}".format(opt.train_batch_size))
    print("Train iterations per epoch: {}".format(len(trainloader)))

    if opt.backbone == "resnet18":
        model = resnet18(pretrained=True)
        model.fc = ResNetFC(512, opt.emb_feat_size)
    elif opt.backbone == "resnet34":
        model = resnet34(pretrained=True)
        model.fc = ResNetFC(512, opt.emb_feat_size)
    elif opt.backbone == "resnet50":
        model = resnet50(pretrained=True)
        model.fc = ResNetFC(512, opt.emb_feat_size)

    if opt.metric == "add_margin":
        metric_fc = AddMarginProduct(
            opt.emb_feat_size, num_classes, device, s=opt.s, m=opt.m)
    elif opt.metric == "arc_margin":
        metric_fc = ArcMarginProduct(
            opt.emb_feat_size, num_classes, device, s=opt.s, m=opt.m)
    elif opt.metric == "sphere":
        metric_fc = SphereProduct(
            opt.emb_feat_size, num_classes, device, m=opt.m)
    else:
        metric_fc = Linear(opt.emb_feat_size, num_classes)

    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    criterion = CrossEntropyLoss()
    if opt.optimizer == "sgd":
        optimizer = SGD([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = Adam([{"params": model.parameters()}, {"params": metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    for epoch in range(opt.max_epoch):
        if epoch > 0:
            scheduler.step()
        for phase in ['train', 'val']:
            print("")
            print("CURRENT PHASE: {}".format(phase))
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)
                
            for ii, data in enumerate(data_loaders[phase]):
                data_input, label = data
                data_input = data_input.to(device)
                label = label.to(device).long()
                feature = model(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)
                optimizer.zero_grad()
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                iters = epoch * len(data_loaders[phase]) + ii

                if iters % opt.print_freq == 0:
                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1)
                    label = label.data.cpu().numpy()
                    acc = np.sum((output == label).astype(int)) / label.shape[0]
                    print("Epoch -- {} Iter -- {} {} Loss -- {:.6f} {} Acc -- {:.6f}".format(
                        epoch, ii, phase, loss.item(), phase, acc))

        if epoch % opt.save_interval == 0 or epoch == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, epoch)

        model.eval()
