import os
import torch


def save_model(model, dataset, mode, metric, backbone):
    # save trained model to cehckpoints directory
    save_path = os.path.join(os.path.abspath(
        ""), "checkpoints", "{}_{}_{}_{}.pth".format(dataset, mode, metric, backbone))
    torch.save(model, save_path)
    return


def load_model(dataset, mode, metric, backbone):
    # load trained model from checkpoints directory
    load_path = os.path.join(os.path.abspath(
        ""), "checkpoints", "{}_{}_{}_{}.pth".format(dataset, mode, metric, backbone))
    model = torch.load(load_path).eval()
    return model
