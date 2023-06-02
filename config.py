import argparse

def parse_option():
    parser = argparse.ArgumentParser('Training Vision Transformer', add_help=False)

    # easy config modification
    parser.add_argument('--gpus',required=True,help='Using GPU 0,1,2,3 or cpu')
    parser.add_argument('--dataset', type=str, default='ImageNet',help='Kind of Dataset')
    parser.add_argument('--data-path', type=str, default='/data/ssd2/ImageNet/ImageNet_train/',help='path to dataset')
    parser.add_argument('--test-data-path', type=str, default='/data/ssd2/ImageNet/ImageNet_val/',help='path to dataset')
    
    parser.add_argument('--epochs',type=int,default=300,help='epoch size to train the model')
    parser.add_argument('--early-stopping',type=int,default=10,help='using early stopping to train the model')
    parser.add_argument('--batch-size', type=int,default=512, help="batch size for single GPU")
    
    # Mixed Precision
    parser.add_argument('--fp16',default=True,
                        help='Using Mixed Precision or Not')
    
    parser.add_argument('--input-size',type=int,default=224,help='size of input image')
    # batch 512 / lr = 0.01 -> nan & 512 / 0.001 -> nan
    parser.add_argument('--lr',type=int,default=0.0001,help='learning rate')
    parser.add_argument('--optim',default='AdamW',help='Kind of optimization to train the model')
    parser.add_argument('--scheduler',default='cosine',help='learning rate scheduler to control the size of learning rate')
    parser.add_argument('--weight-decay',type=float,default=0.1,help='for regularization to train the model')
    
    parser.add_argument('--label-smoothing',type=bool,default=False,help='label smoothing method to train the model more robust')
    parser.add_argument('--smoothing-p',type=float,default=0.,help='ratio of label smoothing')
    parser.add_argument('--grad-clipping',type=bool,default=True,help='gradient clipping method to prevent explorer')
    
    parser.add_argument('--total-train-percent',type=float,default=1.,help='To control the ratio of training dataset')
    parser.add_argument('--train-percent',type=float,default=0.9,help='ratio of training dataset')
    parser.add_argument('--seed',type=int,default=2,help='Value of random seed to implement the representation')
    
    parser.add_argument('--cutmix-p',type=float,default=0.,help='Using cutmix method for data augmentation')
    parser.add_argument('--beta',type=float,default=1.,help='value of beta for cutmix')
    
    # About Transformer
    parser.add_argument('--model-name',default='ViT-B',help='Kind of VisionTransformer (ViT-B, ViT-L, ViT-H)')
    parser.add_argument('--dropout',type=float,default=0.1,help='Dropout ratio')
    parser.add_argument('--emb-dropout',type=float,default=0.,help='Dropout ratio')
    parser.add_argument('--patch-size',type=int,default=16,help='Size of patch for Transformer architecture')
    parser.add_argument('--pool',default='cls',help = 'Choose one from mean and cls')
    
    parser.add_argument('--bandwidth',default=None,help = 'Choose one from mean and cls')
    parser.add_argument('--learnable_mask',default=False,help = 'Choose one from mean and cls')
    args, unparsed = parser.parse_known_args()
    return args



