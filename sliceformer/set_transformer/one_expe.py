import numpy as np
import argparse
import os
from set_transformer.trainer_model_net import main
import wandb
import torch
import random

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='sinkformer', help='sinkformer or transformer')
parser.add_argument("--n_it", type=int, default=1, help='number of iterations within sinkkorn')
parser.add_argument("--num_pts", type=int, default=1000, help='number of points to sample')
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=200)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--visual_attn', type=bool, default=False)
parser.add_argument('--attn', type=str, default='swd', choices=['sink', 'swd'])
args = parser.parse_args()

def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

model = args.model
n_it = args.n_it
num_pts = args.num_pts
learning_rate = args.learning_rate
batch_size = args.batch_size
dim = args.dim
n_heads = args.n_heads
n_anc = args.n_anc
train_epochs = args.train_epochs
seed = args.seed

save_dir = 'results'
try:
    os.mkdir(save_dir)
except:
    pass

checkpoint_dir = 'checkpoint_ModelNet/' + save_dir

try:
    os.makedirs(checkpoint_dir)
except:
    pass
print(num_pts)

# mode: ("online", "offline", "disabled")
mode = 'disabled' if args.visual_attn else 'online'
run = wandb.init(project='SetTransformer', config=args, mode=mode)
if model == 'transformer':
    run.name = 'trans-%s'%(run.id)
else:
    run.name = args.attn + '-%s'%(run.id)


save_adr = save_dir + '/%s_epochs_%d_nit_%d_lr_%f_np_%d_s_%d.npy' % (model, train_epochs, n_it, learning_rate, num_pts, seed)
seed_everything(seed)
res = main(model, n_it, num_pts, learning_rate, batch_size, dim, n_heads, n_anc, train_epochs, save_adr, args.attn, wandb)

