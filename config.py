class Config(object):
    env = "default"
    dataset = "CIFAR10"
    backbone = "resnet18"
    train_mode = "clean"
    test_mode = "pgd_20"  # clean, fgsm, bim, pgd_7, pgd_20

    emb_feat_size = 512  # feature embedding size
    metric = "arcface"  # softmax, arcface, cosface, sphereface
    m = 0.15
    s = 4

    val_split = 0.1
    batch_size = 256
    max_epoch = 600
    patience = 600

    scheduler = "dynamic"  # decay, dynamic, cycle
    optimizer = "sgd"

    # decay scheduler params
    lr = 0.1
    lr_step = 1
    lr_decay = 0.99  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001

    # cycle scheduler params
    cycle_lr = 3e-3
    cycle_factor = 6

    use_gpu = True
    gpu_id = "0"
    num_workers = 0
