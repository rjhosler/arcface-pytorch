import os
import torch


def save_model(model, dataset, metric, mode, backbone):
    # save trained model to cehckpoints directory
    if metric == "bb":
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}.pth".format(dataset, metric, backbone))
    else:
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone))
    torch.save(model, save_path)
    return
    
def save_model_aaml(model, dataset, metric, mode, backbone, s):
    # save trained model to cehckpoints directory
    if metric == "bb":
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}_{}.pth".format(dataset, metric, backbone, s))
    else:
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone, s))
    torch.save(model, save_path)
    return


def load_model(dataset, metric, mode, backbone):
    # load trained model from checkpoints directory
    if metric == "bb":
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}.pth".format(dataset, metric, backbone))
    else:
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone))
    model = torch.load(load_path).eval()
    return model
