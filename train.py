import numpy as np
import torch
from config import Config
from models.train_aaml import train as aaml_train
from models.train_center import train as center_train
from models.train_triplet import train as triplet_train
from models.train_softmax import train as softmax_train
from models.train_contrastive import train as contrastive_train
import os

gpu = Config()
os.environ["CUDA_VISIBLE_DEVICES"]=gpu.GPU
seed = 20 #previously 42.

np.random.seed(seed)
torch.manual_seed(seed)


if __name__ == "__main__":
    # load in arguments defined in config/config.py
    opt = Config()
    
    if opt.metric == "multiple":
        for metric in opt.metrics:
            np.random.seed(seed)
            torch.manual_seed(seed)
            opt.metric = metric
            
            # perform contrastive loss training
            if opt.metric == "contrastive":
                np.random.seed(seed)
                torch.manual_seed(seed)
                contrastive_train(opt)

            # perform triplet loss training
            elif opt.metric == "triplet":
                np.random.seed(seed)
                torch.manual_seed(seed)
                triplet_train(opt)

            # perform center loss training
            elif opt.metric == "center":
                np.random.seed(seed)
                torch.manual_seed(seed)
                center_train(opt)

            # perform aaml loss training
            elif opt.metric == "aaml":
                np.random.seed(seed)
                torch.manual_seed(seed)
                aaml_train(opt)

            # perform softmax loss training
            else:
                opt.test_bb = False
                np.random.seed(seed)
                torch.manual_seed(seed)
                softmax_train(opt)
        
    # perform contrastive loss training
    elif opt.metric == "contrastive":
        np.random.seed(seed)
        torch.manual_seed(seed)
        contrastive_train(opt)

    # perform triplet loss training
    elif opt.metric == "triplet":
        np.random.seed(seed)
        torch.manual_seed(seed)
        triplet_train(opt)

    # perform center loss training
    elif opt.metric == "center":
        np.random.seed(seed)
        torch.manual_seed(seed)
        center_train(opt)

    # perform aaml loss training
    elif opt.metric == "aaml":
        np.random.seed(seed)
        torch.manual_seed(seed)
        aaml_train(opt)

    # perform softmax loss training
    else:
        opt.test_bb = False
        np.random.seed(seed)
        torch.manual_seed(seed)
        softmax_train(opt)

        # perform training of second model (using different seeds) to use for black box attack
        if opt.metric == "softmax" and opt.train_mode == "clean":
            opt.test_bb = True
            np.random.seed(seed)
            torch.manual_seed(seed)
            softmax_train(opt)
