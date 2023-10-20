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

torch.set_printoptions(profile='full', precision=6)


def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def visual_matrix(attention_weights):
    # |attention_weights| : (batch_size, n_heads, seq_len, seq_len)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_weights[0, 0])
    fig.colorbar(cax)
    plt.show()


def visual_atten(attention_weights):
    # |attention_weights| : [epoch x (batch_size, n_heads, seq_len, seq_len)]
    for i in range(0, len(attention_weights), int(len(attention_weights) / 10)):
        col_sorted, col_indices = attention_weights[i][0, 0].sum(-2).sort(-1)
        x = list(range(len(col_sorted)))
        l = plt.plot(x, col_sorted, label=i)
    plt.xlabel('Sorted columns')
    plt.ylabel('Sum of coefficients')
    plt.legend()
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--n_it", type=int, default=5, help='number of iterations within sinkkorn')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--ps", type=int, default=4, help='patch size')
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--channels", type=int, default=3, help='patch size')
parser.add_argument("--padding", type=int, default=4)
parser.add_argument('--size', type=int, default=32)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--n_epochs', type=int, default=400)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--mlp_dim', type=int, default=512)
parser.add_argument('--depth', type=int, default=6)
parser.add_argument('--emb_dropout', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--z', type=int, default=160)
parser.add_argument('--v', type=int, default=512)
parser.add_argument('--h', type=int, default=32)
parser.add_argument('--attn', type=str, default='swd', choices=['trans', 'sink', 'swd'])
parser.add_argument('--visual_attn', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--Dpath', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'FakeData'])
parser.add_argument('--save_ckpt', type=str, default='/home/shen_yuan/slicedformer/vit-pytorch/exps/')
parser.add_argument('--load_ckpt', type=str,
                    default='/home/shen_yuan/slicedformer/vit-pytorch/exps/cifar10/swd/best.ckpt')
parser.add_argument('--model', type=str, default='ema', choices=['ema', 'Haar'])
parser.add_argument('--validate', action='store_true', default=False)
parser.add_argument('--validate_equ', action='store_true', default=False)
args = parser.parse_args()

n_it = args.n_it
seed = args.seed
lr = args.lr
ps = args.ps
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

train_shuffle = False if args.visual_attn else True
if args.Dpath == 'mnist':
    args.size = 28
    args.channels = 1
    args.ps = 2
    args.n_epochs = 100
    args.num_classes = 10
    mean, std = [0.1307], [0.3081]
elif args.Dpath == 'cifar10':
    args.size = 32
    args.channels = 3
    args.ps = 4
    args.n_epochs = 400
    args.num_classes = 10
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
elif args.Dpath == 'cifar100':
    args.size = 32
    args.channels = 3
    args.ps = 2
    args.n_epochs = 400
    args.num_classes = 100
    mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
elif args.Dpath == 'FakeData':
    args.size = 320
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
else:
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

imsize = int(args.size)
transform_train = transforms.Compose([
    transforms.RandomCrop(args.size, padding=args.padding),
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

if args.Dpath == 'mnist':
    tr_set = torchvision.datasets.MNIST(args.Dpath, train=True, download=True, transform=transform_train)
    ts_set = torchvision.datasets.MNIST(args.Dpath, train=False, download=True, transform=transform_test)
elif args.Dpath == 'cifar10':
    tr_set = torchvision.datasets.CIFAR10(args.Dpath, train=True, download=True, transform=transform_train)
    ts_set = torchvision.datasets.CIFAR10(args.Dpath, train=False, download=True, transform=transform_test)
elif args.Dpath == 'cifar100':
    tr_set = torchvision.datasets.CIFAR100(args.Dpath, train=True, download=True, transform=transform_train)
    ts_set = torchvision.datasets.CIFAR100(args.Dpath, train=False, download=True, transform=transform_test)
elif args.Dpath == 'FakeData':
    transform_test = transforms.ToTensor()
    tr_set = torchvision.datasets.FakeData(size=1000, image_size=(3, args.size, args.size), transform=transform_test)
    ts_set = torchvision.datasets.FakeData(size=10000, image_size=(3, args.size, args.size), transform=transform_test)

tr_load = torch.utils.data.DataLoader(tr_set, batch_size=args.batch_size, shuffle=train_shuffle,
                                      num_workers=args.num_workers)
ts_load = torch.utils.data.DataLoader(ts_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


def train_iter(model, optimz, data_load, save_adr):
    samples = len(data_load.dataset)
    model.train()
    csamp = 0
    tloss = 0

    attention_weights = None
    for i, (data, target) in enumerate(data_load):
        data = data.to(device)
        target = target.to(device)
        optimz.zero_grad()
        output, attn_weights = model(data, training=True)
        out = F.log_softmax(output, dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()

        if attention_weights is None and attn_weights is not None:
            attention_weights = attn_weights.cpu()

        tloss += loss.item()
        _, pred = torch.max(output, dim=1)
        csamp += pred.eq(target).sum()
    acc = 100.0 * csamp / samples
    aloss = tloss / samples
    print('Average train loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)')

    return attention_weights, aloss, acc


def evaluate(model, data_load):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0

    with torch.no_grad():
        for data, target in data_load:
            data = data.to(device)
            target = target.to(device)
            output, attn_weights = model(data, training=False)
            out = F.log_softmax(output, dim=1)
            loss = F.nll_loss(out, target, reduction='sum')
            _, pred = torch.max(out, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target).sum()
    acc = 100.0 * csamp / samples
    aloss = tloss / samples
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


def main(N_EPOCHS=45, heads=1, mlp_dim=128, max_iter=n_it, eps=1, lr=lr, depth=1, image_size=32, channels=3,
         attn='trans',
         num_classes=10, dropout=0.1, emb_dropout=0.1, ps=ps, seed=seed, dim=1024, save_adr='results_mnist', wandb=None,
         validate=False, validate_equ=False, load_ckpt='last.ckpt', args=args):
    if validate_equ:
        checkpoint = torch.load(load_ckpt)
        model = ViT_equ(image_size=image_size, patch_size=ps, num_classes=num_classes, channels=channels,
                        emb_dropout=emb_dropout, dropout=dropout, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                        max_iter=max_iter, eps=eps, attn=attn).to(device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        evaluate_equ(model, ts_load)
        return

    if args.model == 'Haar':
        model = ViT_Haar(image_size=image_size, patch_size=ps, num_classes=num_classes, channels=channels,
                         emb_dropout=emb_dropout, dropout=dropout, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                         max_iter=max_iter, eps=eps, attn=attn).to(device)
    elif args.model == 'ema':
        model = ViT_ema(image_size=image_size, patch_size=ps, num_classes=num_classes, channels=channels,
                        emb_dropout=emb_dropout, dropout=dropout, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                        max_iter=max_iter, eps=eps, attn=attn, zdim=args.z, ndim=args.v, hdim=args.v).to(device)
    print('total params:', sum(p.numel() for p in model.parameters()))

    if validate:
        checkpoint = torch.load(load_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate(model, ts_load)
        return

    optimz = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimz, N_EPOCHS)

    attn_weights_array = []
    best_test_acc = 0
    for epoch in range(1, N_EPOCHS + 1):
        # if epoch == 35:
        #     for g in optimz.param_groups:
        #         print('lr /= 10')
        #         g['lr'] /= 10
        # if epoch == 41:
        #     for g in optimz.param_groups:
        #         print('lr /= 10')
        #         g['lr'] /= 10
        print('Epoch:', epoch)
        start_time = time.time()
        attn_weights, train_loss, train_acc = train_iter(model, optimz, tr_load, save_adr)
        if args.visual_attn:
            visual_matrix(attn_weights)
        # attn_weights_array.append(attn_weights)
        # if len(attn_weights_array) % 10 == 0 and len(attn_weights_array) != 0:
        #     visual_atten(attn_weights_array)
        print('finish one epoch in %.4f seconds' % (time.time() - start_time))
        scheduler.step()
        test_loss, test_acc = evaluate(model, ts_load)
        wandb.log({
            'train loss': train_loss,
            'train acc': train_acc,
            'test loss': test_loss,
            'test acc': test_acc,
        })

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(save_adr, 'best.ckpt'))


def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


# mode: ("online", "offline", "disabled")
mode = 'disabled' if args.visual_attn or (not args.wandb) else 'online'

run = wandb.init(project=args.Dpath, config=args, mode=mode)
run.name = args.attn + '-%s' % (run.id)
seed_everything(seed)
main(N_EPOCHS=args.n_epochs, heads=args.n_heads, dim=args.dim, mlp_dim=args.mlp_dim, max_iter=n_it, eps=1, lr=lr,
     depth=args.depth, dropout=args.dropout, emb_dropout=args.emb_dropout, ps=ps, channels=args.channels,
     num_classes=args.num_classes, attn=args.attn, image_size=args.size, seed=seed,
     save_adr=os.path.join(args.save_ckpt, args.Dpath, args.attn), wandb=wandb,
     validate=args.validate, validate_equ=args.validate_equ, load_ckpt=args.load_ckpt, args=args)
