class Config(object):
    env = "default"
    dataset = "CIFAR10"
    backbone = "resnet18"

    emb_feat_size = 512  # feature embedding size
    metric = "softmax" # arcface, cosface, sphereface, softmax
    m = 0.35
    s = 10

    val_split = 0.1
    batch_size = 256
    max_epoch = 100
    patience = 100

    optimizer = "sgd"
    lr = 0.1  # initial learning rate
    lr_step = 2
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
    
    #upper and lower lr for cyclic training
    lr_max = 0.05
    lr_base = 0.005

    use_gpu = True
    gpu_id = "0"
    num_workers = 8
    print_freq = 15
