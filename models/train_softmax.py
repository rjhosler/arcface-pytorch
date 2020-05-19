import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn import CrossEntropyLoss, MSELoss, DataParallel
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from models.utils import save_model, load_model
from models.metrics import Softmax
from models.attacks import pgd
from models.resnet_cifar import resnet18, resnet34


np.random.seed(42)
torch.manual_seed(42)


def train(opt):
    # set device to cpu/gpu
    if opt.use_gpu:
        device = torch.device("cuda", opt.gpu_id)
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

    print("Dataset -- {}, Metric -- {}, Train Mode -- {}, Backbone -- {}".format(opt.dataset,
                                                                                 opt.metric, opt.train_mode, opt.backbone))
    print("Train iteration batch size: {}".format(opt.batch_size))
    print("Train iterations per epoch: {}".format(len(train_loader)))

    # get backbone model
    if opt.backbone == "resnet18":
        model = resnet18(pretrained=False)
    else:
        model = resnet34(pretrained=False)

    model.fc = Softmax(model.fc.in_features, num_classes)

    model.to(device)
    if opt.use_gpu:
        model = DataParallel(model).to(device)

    criterion = CrossEntropyLoss()
    mse_criterion = MSELoss()

    # set optimizer and LR scheduler
    if opt.optimizer == "sgd":
        optimizer = SGD([{"params": model.parameters()}],
                        lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
    else:
        optimizer = Adam([{"params": model.parameters()}],
                         lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == "decay":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
    else:
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

            start_time = time.time()
            for ii, data in enumerate(data_loaders[phase]):
                # load data batch to device
                images, labels = data
                images = images.to(device)
                labels = labels.to(device).long()

                # perform adversarial attack update to images
                if opt.train_mode == "at" or opt.train_mode == "alp":
                    adv_images = pgd(
                        model, images, labels, 8. / 255, 2. / 255, 7)
                else:
                    pass

                # at train mode prediction
                if opt.train_mode == "at":
                    # logits for adversarial images
                    _, adv_predictions = model(adv_images, labels)

                    # get loss
                    loss = criterion(adv_predictions, labels)
                    optimizer.zero_grad()

                    # for result accumulation
                    predictions = adv_predictions

                # alp train mode prediction
                elif opt.train_mode == "alp":
                    # logits for clean and adversarial images
                    _, predictions = model(images, labels)
                    _, adv_predictions = model(adv_images, labels)

                    # get ce loss
                    ce_loss = criterion(adv_predictions, labels)

                    # get alp loss
                    alp_loss = mse_criterion(adv_predictions, predictions)

                    # get overall loss
                    loss = ce_loss + alp_lambda * alp_loss
                    optimizer.zero_grad()

                    # for result accumulation
                    predictions = adv_predictions

                # clean train mode prediction
                else:
                    # logits for clean images
                    _, predictions = model(images, labels)

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
                    end_time = time.time()
                    acc = total_correct / total_examples
                    loss = total_loss / len(data_loaders[phase])
                    print("{}: Epoch -- {} Loss -- {:.6f} Acc -- {:.6f} Time -- {:.6f}sec".format(
                        phase, epoch, loss, acc, end_time - start_time))

                    if phase == "train":
                        loss = total_loss / len(data_loaders[phase])
                        scheduler.step(loss)
                    else:
                        print("")

    # save model after training for opt.epoch
    if opt.test_bb:
        save_model(model, opt.dataset, "bb", "", opt.backbone)
    else:
        save_model(model, opt.dataset, opt.metric,
                   opt.train_mode, opt.backbone)
