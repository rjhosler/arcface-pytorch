class Config(object):
    env = "default"
    dataset = "CIFAR10"  # CIFAR10, CIFAR100, tiny_imagenet
    backbone = "resnet18"  # resnet18, resnet34
    train_mode = "clean"  # clean, at, alp
    test_mode = "fgsm"  # clean, fgsm, bim, cw, pgd_7, pgd_20, mim
    test_restarts = 1  # 1 (for all but 20PGD20), 20
    test_bb = True  # True, False, test using black box attack
    bb_metric = "softmax" # softmax, contrastive, triplet, center, aaml

    #train multiple clean "black box" models consecutively. 
    metrics = ["contrastive", "triplet", "center", "aaml"]  # softmax, contrastive, triplet, center, aaml
    metric = "softmax"

    # arcface margin and scale hyperparameters
    m = 0.2
    s = [2] #[2, 4, 8, 16, 32, 64]

    val_split = 0.1
    batch_size = 256
    epoch = 1000

    scheduler = "dynamic"  # decay, dynamic
    optimizer = "sgd"

    # learning parameters
    lr = 0.1
    lr_step = 100
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0002

    use_gpu = True
    gpu_id = 0
    num_workers = 2
    
   
    GPU = "1"
