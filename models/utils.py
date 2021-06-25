import os
import torch


def save_model(model, dataset, metric, mode, backbone):
    # save trained model to cehckpoints directory
    if metric == "bb":
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "_{}_{}_{}.pth".format(dataset, metric, backbone))
    else:
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "_{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone))
    torch.save(model, save_path)
    return
    
def save_model_aaml(model, dataset, metric, mode, backbone, s, m):
    # save trained model to cehckpoints directory
    if metric == "bb":
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}.pth".format(dataset, metric, backbone))
    else:
        save_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}_{}_{}_m{}.pth".format(dataset, metric, mode, backbone, s, m))
    torch.save(model, save_path)
    return


def load_model(dataset, metric, mode, backbone, s, m):
    # load trained model from checkpoints directory
    if metric == "bb":
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}.pth".format(dataset, metric, backbone))
        print("Test ","{}_{}_{}.pth".format(dataset, metric, backbone))
    elif metric == "aaml":
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}_{}_{}_m{}.pth".format(dataset, metric, mode, backbone, s, m))
        print("{}_{}_{}_{}_{}_m{}.pth".format(dataset, metric, mode, backbone, s, m))
    else:
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone))
        print("{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone))
    model = torch.load(load_path).eval()
    return model
    
def load_model_underscore(dataset, metric, mode, backbone, s, m):
    # load trained model from checkpoints directory
    if metric == "bb":
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "_{}_{}_{}.pth".format(dataset, metric, backbone))
        print("Test ","_{}_{}_{}.pth".format(dataset, metric, backbone))
    elif metric == "aaml":
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "_{}_{}_{}_{}_{}_m{}.pth".format(dataset, metric, mode, backbone, s, m))
        print("_{}_{}_{}_{}_{}_m{}.pth".format(dataset, metric, mode, backbone, s, m))
    else:
        load_path = os.path.join(os.path.abspath(
            ""), "checkpoints", "_{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone))
        print("_{}_{}_{}_{}.pth".format(dataset, metric, mode, backbone))
    model = torch.load(load_path).eval()
    return model
