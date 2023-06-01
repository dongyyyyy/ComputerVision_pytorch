import logging
import os

# model
from models.vit import *
# config
from ImageNet_classes import *
# Customized Dataloader
from util.Dataloader import *

def train_visionTransformer(args,logger,save_filename):
    return

def training_visionTransformer(args):
    dataset_path = args.data_path
    test_dataset_path = args.test_data_path
    
    random_seed = args.seed
    
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    
    if args.dataset == 'ImageNet':
        class_list = os.listdir(dataset_path)
        dataset_path = [dataset_path + class_name + '/' for class_name in class_list]
        test_dataset_path = [test_dataset_path + class_name + '/' for class_name in class_list]
    
    print(len(dataset_path), ' / ', len(test_dataset_path)    )
    
    
    
    dropout = 0.1
    # scaler = GradScaler()
    # max_norm = 1
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
    training_list = []
    validation_list = []
    test_list = []
    
    for class_index in range(len(dataset_path)):
        file_list = os.listdir(dataset_path[class_index])
        file_list.sort()
        
        # split training and validation
        for file_index,filename in enumerate(file_list):
            if file_index < int(len(file_list)* args.train_percent):
                training_list.append(dataset_path[class_index]+filename)
            else:
                validation_list.append(dataset_path[class_index]+filename)

        # Test dataset
        file_list = os.listdir(test_dataset_path[class_index])
        # file_list.sort()

        for file_index,filename in enumerate(file_list):
            test_list.append(test_dataset_path[class_index]+filename)

    # number of samples 
    print(len(training_list), ' / ', len(validation_list), ' / ', len(test_list))
    # 1,281,167
    print(len(training_list) + len(validation_list))

    model_save_path = f'/data/hdd3/Vision_transformer/saved_model/{args.dataset}/'
    logging_save_path = f'/data/hdd3/Vision_transformer/log/{args.dataset}/'
    # model_save_path = '/home/eslab/kdy/git/Sleep_pytorch/saved_model/seoulDataset/single_epoch_models/'
    # logging_save_path = '/home/eslab/kdy/git/Sleep_pytorch/log/seoulDataset/single_epoch_models/'

    os.makedirs(model_save_path,exist_ok=True)
    os.makedirs(logging_save_path,exist_ok=True)
    
    save_filename = model_save_path + f'ViT.pth'
    
    logging_filename = logging_save_path + f'ViT.txt'
    
    
    
    
    
    
    train_first = True
    try:
        check_file = open(logging_filename, 'r')
        # print(check_file)
        checking_finish = False
        while True:
            line = check_file.readline()
            if not line : break
            if line.split(' ')[-1].replace("\n", "") == 'Training_End':
                checking_finish = True
        check_file.close()
        if checking_finish:
            print(f'{logging_filename} had been finish!!!')
            # return
        train_first = False
    except:
        print('This is first Start!!')

    
    ######################## Logging ###################################
    # logger instance
    logger = logging.getLogger(__name__)

    # formatter
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
    
    # create handler (stream, file)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(logging_filename)
    
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)
    if train_first:
        logger.info('Training_Start(First)')
    else:
        logger.info('Training_continue...')
    # logger.info('Training_End')
    
    print('save filename : ',save_filename)
    
    
    train_visionTransformer(args = args,logger = logger,save_filename = save_filename)
    