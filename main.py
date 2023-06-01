import argparse

from train.ViT.train_vit import *

def parse_option():
    parser = argparse.ArgumentParser('Training Vision Transformer', add_help=False)

    # easy config modification
    parser.add_argument('--gpus',required=True,help='Using GPU 0,1,2,3 or cpu')
    parser.add_argument('--dataset', type=str, default='ImageNet',help='Kind of Dataset')
    parser.add_argument('--data-path', type=str, default='/data/ssd2/ImageNet/ImageNet_train/',help='path to dataset')
    parser.add_argument('--test-data-path', type=str, default='/data/ssd2/ImageNet/ImageNet_val/',help='path to dataset')
    
    parser.add_argument('--epochs',type=int,default=100,help='epoch size to train the model')
    parser.add_argument('--early-stopping',type=int,default=5,help='using early stopping to train the model')
    parser.add_argument('--batch-size', type=int,default=256, help="batch size for single GPU")
    
    # Mixed Precision
    parser.add_argument('--fp16',default=False,
                        help='Using Mixed Precision or Not')
    
    parser.add_argument('--lr',type=int,default=0.001,help='learning rate')
    parser.add_argument('--optim',default='AdamW',help='Kind of optimization to train the model')
    parser.add_argument('--scheduler',default='cosine',help='learning rate scheduler to control the size of learning rate')
    parser.add_argument('--weight_decay',type=float,default=0.1,help='for regularization to train the model')
    
    parser.add_argument('--label-smoothing',type=bool,default=False,help='label smoothing method to train the model more robust')
    parser.add_argument('--grad-clipping',type=bool,default=False,help='gradient clipping method to prevent explorer')
    
    parser.add_argument('--total-train-percent',type=float,default=1.,help='To control the ratio of training dataset')
    parser.add_argument('--train-percent',type=float,default=0.9,help='ratio of training dataset')
    parser.add_argument('--seed',type=int,default=2,help='Value of random seed to implement the representation')
    
    
    args, unparsed = parser.parse_known_args()
    return args

'''
    if training_mode == 'pretraining':
        optimizer = 'Adam'
        scheduler = 'linear'
        weight_decay = 0.1
        learning_rate = 0.001
        batch_size = 4096
        using_label_smoothing = True
    elif training_mode == 'finetuning':
        optimizer = 'SGD'
        scheduler = 'cosine'
        weight_decay = None
        # weight_decay = 0.03
        # weight_decay = 0.3
        
        learning_rate = 0.001
        batch_size = 512
        using_grad_clipping = True
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        using_label_smoothing = True
'''

'''
use_dataset='ImageNet',image_size=224,
                               fp16=False,learning_rate=0.001,optimizer = 'Adam',scheduler='cosine',weight_decay = 0.1,batch_size = 512,using_label_smoothing = False,
                               using_grad_clipping = True,
                               train_total_percent=1.,train_percent=0.8,random_seed=2,
                               gpu_num=[0]
'''

if __name__ == '__main__':
    args = parse_option()
        
    training_visionTransformer(args)


# model = ViT(num_classes=1000,image_size=224 ,patch_size=16, dim=768, depth=12, heads=12, mlp_dim=768*4, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.)
# # model = ViT(num_classes=1000,image_size=224 ,patch_size=16, dim=1024, depth=24, heads=16, mlp_dim=1024*4, pool = 'cls', channels = 3, dropout = 0., emb_dropout = 0.)

# summary(model.cuda(),(3,224,224))

