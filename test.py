import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.nn import DataParallel, CrossEntropyLoss
from torch.nn.functional import normalize
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from config import Config
from models.utils import load_model
from models.attacks import fgsm, bim, pgd, mim, cw

os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.random.seed(42)
torch.manual_seed(42)


def test(opt):
    # set device to cpu/gpu
    if opt.use_gpu == True:
        device = torch.device("cuda")
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

    print("Test iteration batch size: {}".format(opt.batch_size))
    print("Test iterations per epoch: {}".format(len(test_loader)))

    model = load_model(opt.dataset, opt.train_mode, opt.metric, opt.backbone)

    model.to(device)
    model = DataParallel(model)

    # get prediction results for model
    y_true, y_pred = [], []
    acc_accum = []
    for ii, data in enumerate(test_loader):
        # load data batch to device
        data_input, label = data

        output = label.cpu().numpy()
        for restart_cnt in range(opt.test_restarts):
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
            elif opt.test_mode == "cw":
                #(model, images, labels, c, kappa, max_iter, learning_rate, device)
                data_input = cw(model, data_input, label, 0.1, 0, 20, 0.01, device, ii)
            else:
                 pass
        
            # normalize input images
            data_input = data_input.to(device)
            label = label.to(device).long()

            # get feature embedding from resnet and prediction
            output_i = model(data_input, label)

            # accumulate test results
            output_i = output_i.data.cpu().numpy()
            output_i = np.argmax(output_i, axis=1)
            label_i = label.data.cpu().numpy()

            output[np.where(output_i != label_i)
                   ] = output_i[np.where(output_i != label_i)]
            
        y_true.append(label.cpu().numpy())
        y_pred.append(output)
        acc_accum.append(output == label.cpu().numpy())
        
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    print(classification_report(y_true, y_pred))

    acc_accum = np.sum(np.concatenate(
        acc_accum).astype(int)) / np.concatenate(acc_accum).astype(int).shape[0]
    print("Accuracy: {}".format(acc_accum))
    return y_true, y_pred


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()

    # perform training using arguments
    y_true, y_pred = test(opt)
