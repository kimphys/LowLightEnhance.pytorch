class args():

    # training args
    epochs = 4000 # "number of training epochs, default is 2"
    save_per_epoch = 1
    batch_size = 16 # "batch size for training/testing, default is 4"
    dataset1 = "./train_dark.txt"
    dataset2 = "./train_bright.txt"
    HEIGHT = 512
    WIDTH = 512
    lr = 1e-4 # "Initial learning rate, default is 0.001"
    lr_step = 2000 # Learning rate is halved in 10 epochs 	
    # resume = "./models/ckpt_70.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
    resume = None
    save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."
    workers = 20

    # For GPU training
    world_size = -1
    rank = -1
    dist_backend = 'nccl'
    gpu = 0,1,2,3
    multiprocessing_distributed = True
    distributed = None

    # For testing
    test_model =  "./models/ckpt_95.pt"
    test_save_dir = "./results/"
    test_img = "./test.txt"
