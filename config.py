class Config(object):
    env = "default"
    dataset = "CIFAR10"
    backbone = "resnet50"

    emb_feat_size = 512  # feature embedding size. 1000 for vgg16 and squeezenet1_0
    metric = "arcface"  # softmax, cosface, sphereface, softmax
    m = 0.15
    s = 4

    val_split = 0.1
    batch_size = 256
    max_epoch = 600
    patience = 5

    scheduler = "cycle"  # decay, cycle
    optimizer = "adam"

    # decay scheduler params
    lr = 0.05
    lr_step = 200
    lr_decay = 0.1  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.001

    # cycle scheduler params
    cycle_lr = 3e-3
    cycle_factor = 6

    use_gpu = True
    gpu_id = "0"
    num_workers = 4
    print_freq = 15
