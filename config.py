class Config(object):
    env = "default"
    dataset = "CIFAR10"
    backbone = "resnet18"  # resnet18, resnet34
    train_mode = "clean"
    test_mode = "cw"  # clean, fgsm, bim, pgd_7, pgd_20, mim, cw
    test_restarts = 1 # 1 (for all but 20PGD col), 20

    metric = "softmax"  # softmax, arcface, cosface, sphereface
    m = 0.15
    s = 4

    val_split = 0.1
    batch_size = 256
    max_epoch = 5000
    patience = 5000

    scheduler = "dynamic"  # decay, dynamic, cycle
    optimizer = "sgd"

    # decay scheduler params
    lr = 0.1
    lr_step = 1
    lr_decay = 0.99  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0002

    use_gpu = True
    gpu_id = "0"
    num_workers = 2
