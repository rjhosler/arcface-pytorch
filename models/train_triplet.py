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
from models.metrics import Softmax, batch_all_triplet_loss
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
        margin = 0.03
        lambda_loss = [2, 0.001]
        train_set = CIFAR10(root="./data", train=True,
                            download=True, transform=transform_train)
        val_set = CIFAR10(root="./data", train=True,
                          download=True, transform=transform_val)
    else:
        alp_lambda = 0.5
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

    print("Dataset -- {}, Metric -- {}, Train Mode -- {}, Backbone -- {}".format(opt.dataset,
                                                                                 opt.metric, opt.train_mode, opt.backbone))
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

                 # at train mode
                if opt.train_mode == "at":
                    # get feature embedding from resnet
                    features, _ = model(images, labels)
                    adv_features, adv_predictions = model(adv_images, labels)

                    # get triplet loss (margin 0.03 CIFAR10/100, 0.01 TinyImageNet from paper)
                    tpl_loss, _, mask = batch_all_triplet_loss(
                        labels, features, margin, adv_embeddings=adv_features)

                    # get adv anchor and clean pos/neg feature norm loss
                    norm = features.mm(features.t()).diag()
                    adv_norm = adv_features.mm(adv_features.t()).diag()
                    norm_loss = adv_norm[mask.nonzero(
                    )[0]] + norm[mask.nonzero()[1]] + norm[mask.nonzero()[2]]
                    norm_loss = torch.sum(norm_loss) / \
                        (mask.nonzero()[0].shape[0] + mask.nonzero()
                         [1].shape[0] + mask.nonzero()[2].shape[0])

                    # get cross-entropy loss (only adv considering anchor examples)
                    adv_anchor_predictions = adv_predictions[np.unique(
                        mask.nonzero()[0])]
                    anchor_labels = labels[np.unique(mask.nonzero()[0])]
                    ce_loss = criterion(adv_anchor_predictions, anchor_labels)

                    # combine cross-entropy loss, triplet loss and feature norm loss using lambda weights
                    loss = ce_loss + lambda_loss[0] * \
                        tpl_loss + lambda_loss[1] * norm_loss
                    optimizer.zero_grad()

                    # for result accumulation
                    predictions = adv_anchor_predictions
                    labels = anchor_labels

                # alp train mode
                elif opt.train_mode == "alp":
                    # get feature embedding from resnet
                    features, predictions = model(images, labels)
                    adv_features, adv_predictions = model(adv_images, labels)

                    # get triplet loss (margin 0.03 CIFAR10/100, 0.01 TinyImageNet from paper)
                    tpl_loss, _, mask = batch_all_triplet_loss(
                        labels, features, margin, adv_embeddings=adv_features)

                    # get adv anchor and clean pos/neg feature norm loss
                    norm = features.mm(features.t()).diag()
                    adv_norm = adv_features.mm(adv_features.t()).diag()
                    norm_loss = adv_norm[mask.nonzero(
                    )[0]] + norm[mask.nonzero()[1]] + norm[mask.nonzero()[2]]
                    norm_loss = torch.sum(norm_loss) / \
                        (mask.nonzero()[0].shape[0] + mask.nonzero()
                         [1].shape[0] + mask.nonzero()[2].shape[0])

                    # get cross-entropy loss (only considering adv anchor examples)
                    anchor_predictions = predictions[np.unique(
                        mask.nonzero()[0])]
                    adv_anchor_predictions = adv_predictions[np.unique(
                        mask.nonzero()[0])]
                    anchor_labels = labels[np.unique(mask.nonzero()[0])]
                    ce_loss = criterion(adv_anchor_predictions, anchor_labels)

                    # get alp loss
                    alp_loss = mse_criterion(
                        adv_anchor_predictions, anchor_predictions)

                    # combine cross-entropy loss, triplet loss and feature norm loss using lambda weights
                    loss = ce_loss + lambda_loss[0] * \
                        tpl_loss + lambda_loss[1] * norm_loss
                    # combine loss with alp loss
                    loss = loss + alp_lambda * alp_loss
                    optimizer.zero_grad()

                    # for result accumulation
                    predictions = adv_anchor_predictions
                    labels = anchor_labels

                # clean train mode
                else:
                    # get feature embedding from resnet
                    features, predictions = model(images, labels)

                    # get triplet loss (margin 0.03 CIFAR10/100, 0.01 TinyImageNet from paper)
                    tpl_loss, _, mask = batch_all_triplet_loss(
                        labels, features, margin)

                    # get feature norm loss
                    norm = features.mm(features.t()).diag()
                    norm_loss = norm[mask.nonzero()[0]] + \
                        norm[mask.nonzero()[1]] + norm[mask.nonzero()[2]]
                    norm_loss = torch.sum(norm_loss) / \
                        (mask.nonzero()[0].shape[0] + mask.nonzero()
                         [1].shape[0] + mask.nonzero()[2].shape[0])

                    # get cross-entropy loss (only considering anchor examples)
                    anchor_predictions = predictions[np.unique(
                        mask.nonzero()[0])]
                    anchor_labels = labels[np.unique(mask.nonzero()[0])]
                    ce_loss = criterion(anchor_predictions, anchor_labels)

                    # combine cross-entropy loss, triplet loss and feature norm loss using lambda weights
                    loss = ce_loss + lambda_loss[0] * \
                        tpl_loss + lambda_loss[1] * norm_loss
                    optimizer.zero_grad()

                    # for result accumulation
                    predictions = anchor_predictions
                    labels = anchor_labels

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
    save_model(model, opt.dataset, opt.metric, opt.train_mode, opt.backbone)
