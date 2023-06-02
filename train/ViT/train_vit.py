import multiprocessing
from multiprocessing import Process, Manager, Pool, Lock

import logging
import os
import time

import torch
from torch.utils.data import DataLoader
# pip install pytorch
from torchsummary import summary
# pip install tqdm
from tqdm import tnrange, tqdm

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# model
from models.vit import *
# config
from ImageNet_classes import *
# Customized Dataloader
from util.Dataloader import *
from util.function import *
from util.loss_fn import *



def train_visionTransformer(args,logger,saved_model_filename,training_set:list,validation_set:list,test_set:list):
    # number of cpu processor
    cpu_num = multiprocessing.cpu_count()
    #dataload Training Dataset
    if args.dataset == 'ImageNet':
        num_class = 1000
        
    # Dataloader Training Dataset
    train_dataset = ImageNet_dataloader(path=training_set, input_size=args.input_size,num_class=num_class,cutmix_p = args.cutmix_p,beta = args.beta,training=True)
    # weights,count = make_weights_for_balanced_classes(training_set)
    # # print(f'weights : {weights}')
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=(cpu_num//4))
    #train_dataloader = DataLoader(dataset=train_dataset,batch_size=10000,sampler=sampler,num_workers=20)

    # Dataload Validation Dataset
    val_dataset = ImageNet_dataloader(path=validation_set, input_size=args.input_size,num_class=num_class,cutmix_p = 0.,beta = 0.,training=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, num_workers=(cpu_num//4))

    # Dataloader Test Dataset
    test_dataset = ImageNet_dataloader(path=test_set, input_size=args.input_size,num_class=num_class,cutmix_p = 0.,beta = 0.,training=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=(cpu_num//4))
    
    # information of vision transformer
    if args.model_name == 'ViT-B':
        dim = 768
        depth = 12
        heads = 12
    elif args.model_name =='ViT-L':
        dim = 1024
        depth = 24
        heads = 16
    elif args.model_name == 'ViT-H':
        dim = 1280
        depth = 32
        heads = 16
        
    mlp_dim = dim * 4
    
    model = ViT(num_classes=num_class,image_size=args.input_size ,patch_size=args.patch_size, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, pool = args.pool, channels = 3, dropout = args.dropout, emb_dropout = args.emb_dropout,
                bandwidth=args.bandwidth,learnable_mask=args.learnable_mask)
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpus != 'cpu' else 'cpu')
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    
    
    
    print(f'device = {device}')
    if str(device) == 'cuda':
        if len(args.gpus) > 1:
            model = nn.DataParallel(model)
    model = model.to(device)
    summary(model, (3, args.input_size,args.input_size))         
    
    
    if args.fp16:
        scaler = GradScaler(enabled=True)
    if args.grad_clipping:
        max_norm = 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # hook_forward = []
    # for name,layer in list(model.named_modules()):
    #     if name.split('.')[-1] == 'attend':
    #         hook_forward.append(Hook(layer,backward=False))
    if args.cutmix_p > 0.:
        loss_fn = CutMixCrossEntropyLoss()
    else:
        if args.label_smoothing:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing_p)
        else:
            loss_fn = nn.CrossEntropyLoss()
    
    if args.optim == 'Adam':
        print('Optimizer : Adam')
        b1 = 0.9
        b2 = 0.999
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(b1, b2))
    elif args.optim == 'RMS':
        print('Optimizer : RMSprop')
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        print('Optimizer : SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5,nesterov=False)
    elif args.optim == 'AdamW':
        print('Optimizer AdamW')
        b1 = 0.9
        b2 = 0.999
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(b1, b2),weight_decay=args.weight_decay)
    elif args.optim == 'nAdam':
        b1 = 0.9
        b2 = 0.999
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, betas=(b1, b2),weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        print('Cosine Scheduler')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max=args.epochs)
    
    best_accuracy = 0.
    stop_count = 0
    check_loss = False
    for epoch in range(args.epochs):
        if scheduler != 'None':
            scheduler.step(epoch)
        train_total_loss = 0.0
        train_total_count = 0
        train_total_data = 0

        val_total_loss = 0.0
        val_total_count = 0
        val_total_data = 0

        start_time = time.time()
        model.train()

        output_str = 'current epoch : %d/%d / current_lr : %f \n' % (epoch+1,args.epochs,optimizer.state_dict()['param_groups'][0]['lr'])
        logger.info(output_str)
        
        with tqdm(train_dataloader,desc='Train',unit='batch') as tepoch:
            for index,data in enumerate(tepoch):
                # {'image':image,'label':label_onethot,'class_name':[class_name,class_name_target]}
                batch_images = data['image'].to(device)
                batch_labels = data['label'].long().to(device)
                
                optimizer.zero_grad()
                if args.fp16:
                    with autocast(enabled=True):
                        # 변화도(Gradient) 매개변수를 0으로 만들고
                        
                        pred = model(batch_images)
                        # 순전파 + 역전파 + 최적화를 한 후
                        
                        loss = loss_fn(pred, batch_labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    pred = model(batch_images)
                    # norm = 0
                    # for parameter in model.parameters():
                    #     norm += torch.norm(parameter, p=norm_square)
                    loss = loss_fn(pred, batch_labels) # + beta * norm
                    loss.backward()
                    optimizer.step()
                
                _, predict = torch.max(pred, 1)
                check_count = (predict == batch_labels).sum().item()

                train_total_loss += loss.item()

                train_total_count += check_count
                train_total_data += len(batch_images)
                accuracy = train_total_count / train_total_data
                tepoch.set_postfix(loss=train_total_loss/(index+1),accuracy=100.*accuracy)

        train_total_loss /= index
        train_accuracy = train_total_count / train_total_data * 100

        output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                    % (epoch + 1, args.epochs, time.time() - start_time, train_total_loss,
                        train_total_count, train_total_data, train_accuracy)
        logger.info(output_str)

        # check validation dataset
        start_time = time.time()
        model.eval()

        with tqdm(val_dataloader,desc='Validation',unit='batch') as tepoch:
            for index,data in enumerate(tepoch):
                batch_images = data['image'].to(device)
                batch_labels = data['label'].long().to(device)

                with torch.no_grad():
                    pred = model(batch_images)

                    loss = loss_fn(pred, batch_labels)

                    # acc
                    _, predict = torch.max(pred, 1)
                    check_count = (predict == batch_labels).sum().item()

                    val_total_loss += loss.item()
                    val_total_count += check_count
                    val_total_data += len(batch_images)
                    accuracy = val_total_count / val_total_data
                    tepoch.set_postfix(loss=val_total_loss/(index+1),accuracy=100.*accuracy)

        val_total_loss /= index
        val_accuracy = val_total_count / val_total_data * 100

        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                    % (epoch + 1, args.epochs, time.time() - start_time, val_total_loss,
                        val_total_count, val_total_data, val_accuracy)
        logger.info(output_str)
        
        if epoch == 0:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save_file = save_filename
            stop_count = 0
            start_time = time.time()
            model.eval()

            with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
                for index,data in enumerate(tepoch):
                    batch_images = data['image'].to(device)
                    batch_labels = data['label'].long().to(device)

                    with torch.no_grad():
                        pred = model(batch_images)

                        loss = loss_fn(pred, batch_labels)

                        # acc
                        _, predict = torch.max(pred, 1)
                        check_count = (predict == batch_labels).sum().item()

                        test_total_count += check_count
                        test_total_data += len(batch_images)
                        accuracy = test_total_count / test_total_data
                        tepoch.set_postfix(accuracy=100.*accuracy)


            test_accuracy = test_total_count / test_total_data * 100
            best_test_accuracy = test_accuracy
            output_str = 'test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, args.epochs, time.time() - start_time,
                            test_total_count, test_total_data, test_accuracy)
            logger.info(output_str)
            torch.save({'model_state_dict':model.module.state_dict() if len(gpu_num) > 1 else model.state_dict(),
                            'epoch' : epoch,
                            'optimizer_sate_dict':optimizer.state_dict(),
                            'train_acc':train_accuracy,
                            'validation_acc':best_accuracy,
                            'test_acc':test_accuracy,
                            'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                            'stop_iter':stop_count}
                            , saved_model_filename)
        else:
            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                save_file = save_filename
                
                stop_count = 0
                test_total_count = 0
                test_total_data = 0
                # check validation dataset
                start_time = time.time()
                model.eval()

                with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
                    for index,data in enumerate(tepoch):
                        batch_images = data['image'].to(device)
                        batch_labels = data['label'].long().to(device)

                        with torch.no_grad():
                            pred = model(batch_images)

                            loss = loss_fn(pred, batch_labels)

                            # acc
                            _, predict = torch.max(pred, 1)
                            check_count = (predict == batch_labels).sum().item()

                            test_total_count += check_count
                            test_total_data += len(batch_images)
                            accuracy = test_total_count / test_total_data
                            tepoch.set_postfix(accuracy=100.*accuracy)


                test_accuracy = test_total_count / test_total_data * 100
                best_test_accuracy = test_accuracy
                output_str = 'test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                            % (epoch + 1, args.epochs, time.time() - start_time,
                                test_total_count, test_total_data, test_accuracy)
                logger.info(output_str)
                torch.save({'model_state_dict':model.module.state_dict() if len(gpu_num) > 1 else model.state_dict(),
                            'epoch' : epoch,
                            'optimizer_sate_dict':optimizer.state_dict(),
                            'train_acc':train_accuracy,
                            'validation_acc':best_accuracy,
                            'test_acc':test_accuracy,
                            'learning_rate':optimizer.state_dict()['param_groups'][0]['lr'],
                            'stop_iter':stop_count}, saved_model_filename)
            else:
                stop_count += 1
        if stop_count > stop_iter:
            print('Early Stopping')
            break
        
        output_str = 'best epoch : %d/%d / test accuracy : %f%%\n' \
                    % (best_epoch + 1, args.epochs, best_test_accuracy)
        logger.info(output_str)
        

    output_str = 'best epoch : %d/%d / accuracy : %f%%\n' \
                 % (best_epoch + 1, args.epochs, best_test_accuracy)
    logger.info(output_str)
    logger.info('Training_End')

    check_file.close()
    
    
    '''
    def module_hook(module, grad_input, grad_output):
        answer.extend(grad_input)
        answer.append(grad_output[0])
        pass

    model.register_full_backward_hook(module_hook)
    '''
    
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
    

        
    training_list = []
    validation_list = []
    test_list = []
    
    for class_index in range(len(dataset_path)):
        file_list = os.listdir(dataset_path[class_index])
        file_list.sort()
        
        # split training and validation
        for file_index,filename in enumerate(file_list):
            if file_index < int(int(len(file_list)* args.train_percent) * args.total_train_percent):
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

    model_save_path = f'/data/hdd3/Vision_transformer/saved_model/{args.dataset}_{args.total_train_percent}_{args.train_percent}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}_{args.weight_decay}/random_seed_{args.seed}/'
    logging_save_path = f'/data/hdd3/Vision_transformer/log/{args.dataset}_{args.total_train_percent}_{args.train_percent}/{args.batch_size}_{args.optim}_{args.lr}_{args.scheduler}_{args.weight_decay}/random_seed_{args.seed}/{args.dropout}_{args.emb_dropout}/'
    # model_save_path = '/home/eslab/kdy/git/Sleep_pytorch/saved_model/seoulDataset/single_epoch_models/'
    # logging_save_path = '/home/eslab/kdy/git/Sleep_pytorch/log/seoulDataset/single_epoch_models/'

    os.makedirs(model_save_path,exist_ok=True)
    os.makedirs(logging_save_path,exist_ok=True)
    
    save_filename = model_save_path + f'{args.model_name}({args.pool})_{args.input_size}({args.patch_size})_{args.epochs}_{args.early_stopping}_labelsmoothing_{args.label_smoothing}_{args.smoothing_p}_gradClipping_{args.grad_clipping}_fp16_{args.fp16}.pth'
    
    logging_filename = logging_save_path + f'{args.model_name}({args.pool})_{args.input_size}({args.patch_size})_{args.epochs}_{args.early_stopping}_labelsmoothing_{args.label_smoothing}_{args.smoothing_p}_gradClipping_{args.grad_clipping}_fp16_{args.fp16}.log'
    
    
    
    
    
    
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
    
    
    train_visionTransformer(args = args,logger = logger,saved_model_filename = save_filename,
                            training_set = training_list,validation_set = validation_list,test_set = test_list)
    