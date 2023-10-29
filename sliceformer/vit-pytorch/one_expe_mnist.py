# You need to install the following python packages
# pytorch, vit_pytorch.
import torch
import torchvision
import torchvision.transforms as transforms
from vit_pytorch import ViT, ViT_equ, ViT_Haar, ViT_ema
import time
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import wandb
import os

from utils import create_logger, seed_everything, save_checkpoint, load_state_dict, visual_matrix, visual_atten
from config import config

torch.set_printoptions(profile='full', precision=6)

args = config

device = args['gpu'] if torch.cuda.is_available() else 'cpu'

train_shuffle = False if args['visual_attn'] else True
if args['dataset_name'] == 'mnist':
    args['size'] = 28
    args['channels'] = 1
    args['ps'] = 2
    args['n_epochs'] = 100
    args['num_classes'] = 10
    mean, std = [0.1307], [0.3081]
elif args['dataset_name'] == 'cifar10':
    args['size'] = 32
    args['channels'] = 3
    args['ps'] = 4
    args['n_epochs'] = 100
    args['num_classes'] = 10
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
elif args['dataset_name'] == 'cifar100':
    args['size'] = 32
    args['channels'] = 3
    args['ps'] = 2
    args['n_epochs'] = 100
    args['num_classes'] = 100
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
elif args['dataset_name'] == 'FakeData':
    args['size'] = 320
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
else:
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

imsize = int(args['size'])
transform_train = transforms.Compose([
    transforms.RandomCrop(args['size'], padding=args['padding']),
    transforms.Resize(imsize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if args['dataset_name'] == 'mnist':
    tr_set = torchvision.datasets.MNIST(args['dataset_name'], train=True, download=True, transform=transform_train)
    ts_set = torchvision.datasets.MNIST(args['dataset_name'], train=False, download=True, transform=transform_test)
elif args['dataset_name'] == 'cifar10':
    tr_set = torchvision.datasets.CIFAR10(args['dataset_name'], train=True, download=True, transform=transform_train)
    ts_set = torchvision.datasets.CIFAR10(args['dataset_name'], train=False, download=True, transform=transform_test)
elif args['dataset_name'] == 'cifar100':
    tr_set = torchvision.datasets.CIFAR100(args['dataset_name'], train=True, download=True, transform=transform_train)
    ts_set = torchvision.datasets.CIFAR100(args['dataset_name'], train=False, download=True, transform=transform_test)
elif args['dataset_name'] == 'FakeData':
    transform_test = transforms.ToTensor()
    tr_set = torchvision.datasets.FakeData(size=1000, image_size=(3, args['size'], args['size']), transform=transform_test)
    ts_set = torchvision.datasets.FakeData(size=10000, image_size=(3, args['size'], args['size']), transform=transform_test)

tr_load = torch.utils.data.DataLoader(tr_set, batch_size=args['batch_size'], shuffle=train_shuffle,
                                      num_workers=args['num_workers'])
ts_load = torch.utils.data.DataLoader(ts_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])


def train_iter(model, optimz, data_load, logger, args=None):
    samples = len(data_load.dataset)
    model.train()
    csamp = 0
    tloss = 0

    for i, (data, target) in enumerate(data_load):
        data = data.to(device)
        target = target.to(device)
        optimz.zero_grad()
        output, attn_weights = model(data, training=True)
        out = F.log_softmax(output, dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()

        tloss += loss.item()
        _, pred = torch.max(output, dim=1)
        csamp += pred.eq(target).sum()
    acc = 100.0 * csamp / samples
    aloss = tloss / samples
    logger.info('Average train loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)')
    # print('Average train loss: ' + '{:.4f}'.format(aloss) +
    #       '  Accuracy:' + '{:5}'.format(csamp) + '/' +
    #       '{:5}'.format(samples) + ' (' +
    #       '{:4.2f}'.format(acc) + '%)')

    return aloss, acc


def evaluate(model, data_load, logger=None):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for idx, (data, target) in enumerate(data_load):
            # if idx <20:
            #     continue
            data = data.to(device)
            target = target.to(device)
            output, attn_weights = model(data, training=False)
            out = F.log_softmax(output, dim=1)
            loss = F.nll_loss(out, target, reduction='sum')
            _, pred = torch.max(out, dim=1)
            
            if args['visual_attn']:
                visual_matrix(attn_weights, save_fig='./pv.png')
                1/0

            tloss += loss.item()
            csamp += pred.eq(target).sum()
    acc = 100.0 * csamp / samples
    aloss = tloss / samples
    if logger is not None:
        logger.info('Average test loss: ' + '{:.4f}'.format(aloss) +
            '  Accuracy:' + '{:5}'.format(csamp) + '/' +
            '{:5}'.format(samples) + ' (' +
            '{:4.2f}'.format(acc) + '%)\n')
    else:
        print('Average test loss: ' + '{:.4f}'.format(aloss) +
            '  Accuracy:' + '{:5}'.format(csamp) + '/' +
            '{:5}'.format(samples) + ' (' +
            '{:4.2f}'.format(acc) + '%)\n')
    return aloss, acc


def draw_errorbar(mean, std):
    x = np.arange(len(mean))
    plt.errorbar(x, mean, std, fmt='.k')
    plt.show()


def evaluate_equ(model, data_load):
    model.eval()

    total_cos_sim = []
    total_MAE, total_prob = [], []

    with torch.no_grad():
        for data, target in data_load:
            data = data.to(device)
            # cos_sim = model(data)
            # total_cos_sim.append(cos_sim.unsqueeze(0))
            MAE, prob = model(data)
            total_MAE.append(MAE.unsqueeze(0))
            total_prob.append(prob.unsqueeze(0))
            if len(total_MAE) > 10:
                break

    # total_cos_sim = torch.cat(total_cos_sim, dim=0)
    # mean = total_cos_sim.mean(dim=0)
    # std = total_cos_sim.std(dim=0)
    # print('Mean:')
    # print(mean)
    # print('Std:')
    # print(std)
    # draw_errorbar(mean.cpu().detach().numpy(), std.cpu().detach().numpy())

    total_MAE = torch.cat(total_MAE, dim=0)
    mean = total_MAE.mean(dim=0)
    std = total_MAE.std(dim=0)
    print('Mean:')
    print(mean)
    print('Std:')
    print(std)
    # draw_errorbar(mean.cpu().detach().numpy(), std.cpu().detach().numpy())

    total_prob = torch.cat(total_prob, dim=0)
    mean = total_prob.mean(dim=0)
    std = total_prob.std(dim=0)
    print('Mean:')
    print(mean)
    print('Std:')
    print(std)
    # draw_errorbar(mean.cpu().detach().numpy(), std.cpu().detach().numpy())


def main(args=args):
    if args['validate_equ']:
        checkpoint = torch.load(args['load_checkpoint_path'])
        model = ViT_equ(image_size=args['size'], patch_size=args['ps'], num_classes=args['num_classes'], channels=args['channels'],
                        emb_dropout=args['emb_dropout'], dropout=args['dropout'], dim=args['dim'], depth=args['n_layers'], heads=args['n_heads'], mlp_dim=args['mlp_dim'],
                        max_iter=args['n_it'], eps=1, attn=args['attn']).to(device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        evaluate_equ(model, ts_load)
        return

    if args['model'] == 'vit':
        model = ViT(image_size=args['size'], patch_size=args['ps'], num_classes=args['num_classes'], channels=args['channels'],
                         emb_dropout=args['emb_dropout'], dropout=args['dropout'], dim=args['dim'], depth=args['n_layers'], heads=args['n_heads'], mlp_dim=args['mlp_dim'],
                         max_iter=args['n_it'], eps=1, attn=args['attn']).to(device)
    elif args['model'] == 'Haar':
        model = ViT_Haar(image_size=args['size'], patch_size=args['ps'], num_classes=args['num_classes'], channels=args['channels'],
                         emb_dropout=args['emb_dropout'], dropout=args['dropout'], dim=args['dim'], depth=args['n_layers'], heads=args['n_heads'], mlp_dim=args['mlp_dim'],
                         max_iter=args['n_it'], eps=1, attn=args['attn']).to(device)
    elif args['model'] == 'ema':
        model = ViT_ema(image_size=args['size'], patch_size=args['ps'], num_classes=args['num_classes'], channels=args['channels'],
                        emb_dropout=args['emb_dropout'], dropout=args['dropout'], dim=args['dim'], depth=args['n_layers'], heads=args['n_heads'], mlp_dim=args['mlp_dim'],
                        max_iter=args['n_it'], eps=1, attn=args['attn'], zdim=args['z'], ndim=args['v'], hdim=args['v']).to(device)
    
    if args['validate']:
        model = load_state_dict(os.path.join(args['output_path'], args['dataset_name'], "models", args['load_checkpoint_path'], 'best_model.pth'), model)
        evaluate(model, ts_load)
        return
    
    logger, log_file, exp_id = create_logger(args)
    save_checkpoint_path = os.path.join(args['output_path'], args['dataset_name'], "models", exp_id)
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)
        
    logger.info('total params: {}'.format(sum(p.numel() for p in model.parameters())))
    # print('total params:', sum(p.numel() for p in model.parameters()))

    optimz = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimz, args['n_epochs'])

    attn_weights_array = []
    best_test_acc = 0
    for epoch in range(1, args['n_epochs'] + 1):
        # if epoch == 35:
        #     for g in optimz.param_groups:
        #         print('lr /= 10')
        #         g['lr'] /= 10
        # if epoch == 41:
        #     for g in optimz.param_groups:
        #         print('lr /= 10')
        #         g['lr'] /= 10
        logger.info('Epoch: %d' % epoch)
        # print('Epoch:', epoch)
        start_time = time.time()
        train_loss, train_acc = train_iter(model, optimz, tr_load, logger, args)
        logger.info('finish one epoch in %.4f seconds' % (time.time() - start_time))
        # print('finish one epoch in %.4f seconds' % (time.time() - start_time))
        scheduler.step()
        test_loss, test_acc = evaluate(model, ts_load, logger)
        wandb.log({
            'train loss': train_loss,
            'train acc': train_acc,
            'test loss': test_loss,
            'test acc': test_acc,
        })

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            # torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_checkpoint_path, 'best.ckpt'))
            save_checkpoint(model, args, save_checkpoint_path)


if __name__ == '__main__':
    # mode: ("online", "offline", "disabled")
    mode = 'disabled' if args['visual_attn'] or args['validate'] or (not args['wandb']) else 'online'

    run = wandb.init(project=args['dataset_name'], config=args, mode=mode)
    run.name = args['attn'] + '-%s' % (run.id)
    seed_everything(args['seed'])
    main(args=args)
