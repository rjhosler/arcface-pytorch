class Config(object):
    env = "default"
    dataset = "CIFAR10"
    backbone = "resnet18"

    emb_feat_size = 512  # feature embedding size
    metric = "arc_margin"
    m = 0.35
    s = 4

    checkpoints_path = "checkpoints"
    model_path = "models/{}.pth".format(backbone)
    save_interval = 10

    train_batch_size = 180
    test_batch_size = 64

    input_shape = (1, 128, 128)

    optimizer = "sgd"

    use_gpu = True
    gpu_id = "0"
    num_workers = 8
    print_freq = 5

    debug_file = "/tmp/debug"
    result_file = "result.csv"

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
