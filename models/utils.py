import os
import torch
from torch.nn import Module, Linear


# customize final fully-connected layer of library resnet model
class resnet_fc(Module):
    def __init__(self, out_features):
        super(resnet_fc, self).__init__()
        self.in_features = 512
        self.out_features = out_features

    def forward(self, x):
        if self.in_features == self.out_features:
            return x
        else:
            return Linear(in_features=self.in_features, out_features=self.out_features)


def save_model(model, dataset, metric, backbone):
    # save trained model to cehckpoints directory
    save_path = os.path.join(os.path.abspath(
        ""), "checkpoints", "{}_{}_{}.pth".format(dataset, metric, backbone))
    torch.save(model, save_path)
    return


def load_model(dataset, metric, backbone):
    # load trained model from checkpoints directory
    load_path = os.path.join(os.path.abspath(
        ""), "checkpoints", "{}_{}_{}.pth".format(dataset, metric, backbone))
    model = torch.load(load_path).eval()
    return model
