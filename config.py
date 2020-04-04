class Config(object):
    env = "default"
    dataset = "CIFAR10"
    backbone = "resnet18"

    emb_feat_size = 512 # feature embedding size. 1000 for vgg16 and squeezenet1_0
    metric = "softmax" # arcface, cosface, sphereface, softmax
    m = 0.35
    s = 10

    val_split = 0.1
    batch_size = 256
    max_epoch = 600
    patience = 600

    scheduler = 'StepLR'
    optimizer = "sgd"
    lr = 0.05  # initial learning rate
    lr_step = 200
    lr_decay = 0.1  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.001  
 
    #upper and lower lr for cyclic training
    lr_max = 0.01
    lr_base = 0.001
    step_up = 1
    step_down = 1

    use_gpu = True
    gpu_id = "0"
    num_workers = 16
    print_freq = 15
