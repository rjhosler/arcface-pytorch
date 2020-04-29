class Config(object):
    env = "default"
    dataset = "CIFAR10"  # CIFAR10, CIFAR100, tiny_imagenet
    backbone = "resnet18"  # resnet18, resnet34
    train_mode = "clean"
    test_mode = "clean"  # clean, fgsm, bim, pgd_7, pgd_20, mim, cw
    test_restarts = 1  # 1 (for all but 20PGD col), 20
    test_bb = False  # True, False, test using black box attack

    metric = "arcface"  # softmax, arcface, cosface, sphereface
    m = 0.15
    s = 4

    val_split = 0.1
    batch_size = 256
    max_epoch = 500
    patience = 500

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
