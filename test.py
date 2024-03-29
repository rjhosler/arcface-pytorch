import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import load_model, load_model_underscore
from models.attacks import fgsm, bim, pgd, mim
import os

gpu = Config()
os.environ["CUDA_VISIBLE_DEVICES"]=gpu.GPU


np.random.seed(42)
torch.manual_seed(42)


def test(opt):
    # set device to cpu/gpu
    if opt.use_gpu:
        device = torch.device("cuda", opt.gpu_id)
    else:
        device = torch.device("cpu")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
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

    print("Dataset -- {}, Metric -- {}, Train Mode -- {}, Test Mode -- {}, Blackbox -- {}, Backbone -- {}".format(opt.dataset,
                                                                                                                  opt.metric, opt.train_mode, opt.test_mode, opt.test_bb, opt.backbone))
    print("Test iteration batch size: {}".format(opt.batch_size))
    print("Test iterations per epoch: {}".format(len(test_loader)))

    model = load_model(opt.dataset, opt.metric, opt.train_mode, opt.backbone, opt.s, opt.m)
    model.to(device)
    if opt.use_gpu:
        model = DataParallel(model).to(device)

    # load balck box model for black box attacks
    if opt.test_bb:
        # Test Black box attacks for different metrics
        if opt.bb_metric != "softmax": 
            attack_model = load_model_underscore(opt.dataset, opt.bb_metric, "clean", opt.backbone, opt.s, opt.m) #I trained other bb models in at mode for some reason
            model.to(device)
            if opt.use_gpu:
                model = DataParallel(model).to(device)
        else:
            model_bb = load_model(opt.dataset, "bb", "", opt.backbone, opt.s, opt.m)
            model_bb.to(device)
            if opt.use_gpu:
                model_bb = DataParallel(model_bb).to(device)
            attack_model = model_bb
    else:
        attack_model = model

    # get prediction results for model
    y_true, y_pred = [], []
    for ii, data in enumerate(test_loader):
        # load data batch to device
        images, labels = data
        images = images.to(device)
        labels = labels.to(device).long()
        predictions = labels.cpu().numpy()

        # random restarts for pgd attack
        for restart_cnt in range(opt.test_restarts):
            #print("Batch {}/{} -- Restart {}/{}\t\t\t\t".format(ii+1,len(test_loader), restart_cnt+1, opt.test_restarts))

            # perform adversarial attack update to images
            if opt.test_mode == "fgsm":
                adv_images = fgsm(
                    attack_model, images, labels, 8. / 255)
            elif opt.test_mode == "bim":
                adv_images = bim(
                    attack_model, images, labels, 8. / 255, 2. / 255, 7)
            elif opt.test_mode == "pgd_7":
                adv_images = pgd(
                    attack_model, images, labels, 8. / 255, 2. / 255, 7)
            elif opt.test_mode == "pgd_20":
                adv_images = pgd(
                    attack_model, images, labels, 8. / 255, 2. / 255, 20)
            elif opt.test_mode == "mim":
                adv_images = mim(
                    attack_model, images, labels, 8. / 255, 2. / 255, 0.9, 40)
            else:
                adv_images = images

            # get feature embedding from resnet and prediction
            _, predictions_i = model(adv_images, labels)

            # accumulate test results
            predictions_i = torch.argmax(predictions_i, 1).cpu().numpy()
            labels_i = labels.cpu().numpy()
            predictions[np.where(predictions_i != labels_i)
                        ] = predictions_i[np.where(predictions_i != labels_i)]

        y_true.append(labels.cpu().numpy())
        y_pred.append(predictions)

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    print(classification_report(y_true, y_pred))
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    return y_true, y_pred


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()
    opt.s = 2

    # perform training using arguments
    y_true, y_pred = test(opt)
