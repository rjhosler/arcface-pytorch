class Config(object):
    env = "default"
    dataset = "CIFAR10"
    backbone = "resnet18"

    emb_feat_size = 512  # feature embedding size
    metric = "softmax" # arcface, cosface, sphereface, softmax
    m = 0.35
    s = 4

    val_split = 0.1
    batch_size = 128
    max_epoch = 50
    patience = 5

    optimizer = "sgd"
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    use_gpu = True
    gpu_id = "0"
    num_workers = 4
    print_freq = 15
