import torch
import torch.nn.functional as F

config = {
    # vit trans cifar10_0_2023-10-08-13-45-59-918930
    # vit swd cifar10_0_2023-10-08-12-13-50-925465
    # haar swd  cifar10_0_2023-10-13-15-54-32-123456
    # haar swd epoch300 cifar10_0_2023-10-24-00-08-10-835235
    # haar swd epoch300 cifar100_0_2023-10-24-00-10-24-425483
    # vit swd max cifar10_0_2023-10-23-23-51-14-446800
    # vit swd max epoch300 cifar10_0_2023-10-24-15-42-44-858065
    # vit swd max epoch300 cifar100_0_2023-10-24-15-44-31-422650
    'load_checkpoint_path': '',
    'output_path': '/home/shen_yuan/slicedformer/vit-pytorch/output/',
    
    'wandb': True,
    'validate': False,
    'visual_attn': False,
    'validate_equ': False,
    
    'gpu': 'cuda:4',
    'num_workers': 8,
    'seed': 0,
    
    'attn': 'swd', #choices=['trans', 'sink', 'swd']
    'model': 'vit', # choices=['vit', 'ema', 'Haar']
    
    'n_epochs': 100,
    'batch_size': 64,
    'lr': 0.0001,
    
    'n_layers': 6,
    'n_heads': 8,  
    'dim': 512,
    'mlp_dim': 512,
    'emb_dropout': 0.1,
    'dropout': 0.1,
    
    'dataset_name': 'cifar10', # choices=['mnist', 'cifar10', 'cifar100', 'FakeData']
    'size': 32,
    'ps': 4,
    'num_classes': 10,
    'channels': 3,
    'padding': 4,
    
    'n_it': 5, # number of iterations within sinkkorn
    'z': 160,
    'v': 512,
    'h': 32,
}
