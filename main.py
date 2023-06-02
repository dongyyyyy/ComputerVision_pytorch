from config import parse_option

from train.ViT.train_vit import *


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

