class Config(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 5013
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    machine = 'macos'

    if machine == 'macos':
        train_root = '/Users/bccca/dev/dat/personai_icartoonface_rectrain/icartoonface_rectrain'
        train_list = '/Users/bccca/dev/dat/personai_icartoonface_rectrain/train.txt'
        val_list = '/Users/bccca/dev/dat/personai_icartoonface_rectrain/val.txt'

        checkpoints_path = '/Users/bccca/dev/dat/arcface-pytorch-checkpoints'

    else:
        train_root = '/root/face-rnd/dat/personai_icartoonface_rectrain/icartoonface_rectrain'
        train_list = '/root/face-rnd/dat/personai_icartoonface_rectrain/train.txt'
        val_list = '/root/face-rnd/dat/personai_icartoonface_rectrain/val.txt'

        checkpoints_path = '/root/face-rnd/dat/arcface-pytorch-checkpoints'


    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/data/Datasets/lfw/lfw-align-128'
    lfw_test_list = '/data/Datasets/lfw/lfw_test_pair.txt'

    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 10

    train_batch_size = 16  # batch size
    test_batch_size = 60

    input_shape = (3, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
