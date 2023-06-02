import torch
import numpy as np
def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def save_model(logger,save_filename, model,epoch,optimizer,loss,acc,init_lr,batch_size,scheduler,fp16,seed):
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model_to_save.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : loss,
        'accuracy' : acc,
        'init_lr' : init_lr,
        'batch_size' : batch_size,
        'scheduler' : scheduler,
        'fp16' : fp16,
        'seed' : seed
        }, save_filename)
    logger.info("Saved model checkpoint to [DIR: %s] // loss = %f// accuracy = %f ", save_filename,loss,accuracy)
    
    
    
## Hook
class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self,module,input,output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
