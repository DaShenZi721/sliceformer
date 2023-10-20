import argparse
import os
import trainer_cats_and_dogs
import torch
import random
import numpy as np
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--n_it", type=int, default='3')
parser.add_argument("--seed", type=int, default='2')
parser.add_argument('--attn', type=str, default='trans', choices=['trans', 'sink', 'swd'])
parser.add_argument('--visual_attn', type=bool, default=False)
args = parser.parse_args()


n_it = args.n_it
seed = args.seed

save_dir = 'results'
save_model_dir = 'results_model'

try:
    os.mkdir(save_dir)

except:
    pass

try:
    os.mkdir(save_model_dir)

except:
    pass

def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

mode = 'disabled' if args.visual_attn else 'online'
run = wandb.init(project='dogcat', config=args, mode=mode)
run.name = args.attn + '-%s'%(run.id)

seed_everything(seed)
save_adr = save_dir + '/%d_it_%d.npy' % (n_it, seed)
save_model = save_model_dir + '/%d_it_%d.pth' % (n_it, seed)
res = trainer_cats_and_dogs.main(n_it, save_adr, save_model, seed=seed, wandb=wandb, attn=args.attn)

