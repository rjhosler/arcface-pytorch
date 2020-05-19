import numpy as np
import torch
from config import Config
from models.train_center import train as center_train
from models.train_triplet import train as triplet_train
from models.train_softmax import train as softmax_train
from models.train_contrastive import train as contrastive_train


np.random.seed(42)
torch.manual_seed(42)


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()

    # perform contrastive loss training
    if opt.metric == "contrastive":
        np.random.seed(42)
        torch.manual_seed(42)
        contrastive_train(opt)

    # perform triplet loss training
    elif opt.metric == "triplet":
        np.random.seed(42)
        torch.manual_seed(42)
        triplet_train(opt)

    # perform center loss training
    elif opt.metric == "center":
        np.random.seed(42)
        torch.manual_seed(42)
        center_train(opt)

    # perform softmax/arcface loss training
    else:
        opt.test_bb = False
        np.random.seed(42)
        torch.manual_seed(42)
        softmax_train(opt)

        # perform training of second model (using different seeds) to use for black box attack
        if opt.metric == "softmax" and opt.train_mode == "clean":
            opt.test_bb = True
            np.random.seed(24)
            torch.manual_seed(24)
            softmax_train(opt)
