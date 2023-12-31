import os
import argparse
from prenlp.tokenizer import NLTKMosesTokenizer
from torch.utils.data import DataLoader
import numpy as np
from data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
from trainer import Trainer
import torch
import random
import matplotlib.pyplot as plt
import wandb

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

def visual_matrix(attention_weights):
    # |attention_weights| : (batch_size, n_heads, seq_len, seq_len)
    plt.matshow(attention_weights[0,0])
    plt.show()

def visual_atten(attention_weights):
    # |attention_weights| : [epoch x (batch_size, n_heads, seq_len, seq_len)]
    for i in range(len(attention_weights)):
        col_sorted, col_indices = attention_weights[i][0,0].sum(-2).sort(-1)
        x = list(range(len(col_sorted)))
        l = plt.plot(x, col_sorted, label=i)
    plt.xlabel('Sorted columns')
    plt.ylabel('Sum of coefficients')
    plt.legend()
    plt.show()



def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    print(args)
    save_dir = args.save_dir
    n_it = args.n_it
    seed = args.seed
    save_adr = save_dir + '/%d_it_%d.npy' % (n_it, seed)

    mode = 'disabled' if args.visual_attn else 'online'
    run = wandb.init(project='text classfication', config=args, mode=mode)
    run.name = args.attn + '-%s'%(run.id)

    seed_everything(seed)

    # Load tokenizer
    if args.tokenizer == 'sentencepiece':
        tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_model, vocab_file=args.vocab_file)
    else:
        tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
        tokenizer = Tokenizer(tokenizer=tokenizer, vocab_file=args.vocab_file)
    # Build DataLoader
    train_dataset = create_examples(args, tokenizer, mode='train')
    test_dataset = create_examples(args, tokenizer, mode='test')
    train_test_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    train_dataset, test_dataset = torch.utils.data.random_split(train_test_dataset, [40000, 10000])
    print('train dataset of size %d' % len(train_dataset))
    print('test dataset of size %d' % len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Build Trainer
    trainer = Trainer(args, train_loader, test_loader, tokenizer, n_it=n_it)
    val_loss_array = []
    train_loss_array = []
    val_accuracy_array = []
    train_accuracy_array = []
    attn_weights_array = []
    # Train & Validate
    for epoch in range(1, args.epochs+1):
        epoch_loss, epoch_accuracy, attention_weights_cpu = trainer.train(epoch)
        epoch_val_loss, epoch_val_accuracy = trainer.validate(epoch)
        trainer.save(epoch, args.output_model_prefix)
        val_accuracy_array.append(epoch_val_accuracy)
        train_accuracy_array.append(epoch_accuracy)
        val_loss_array.append(epoch_val_loss)
        train_loss_array.append(epoch_loss)
        trainer.save(epoch, args.output_model_prefix)
        losses = np.asarray([train_loss_array, val_loss_array, train_accuracy_array, val_accuracy_array])
        np.save(save_adr, losses)
        wandb.log({
            'train loss': epoch_loss,
            'train acc': epoch_accuracy,
            'test loss': epoch_val_loss,
            'test acc': epoch_val_accuracy,
        })
        if args.visual_attn:
            attn_weights_array.append(attention_weights_cpu)
            visual_matrix(attention_weights_cpu)
    if args.visual_attn:
        visual_atten(attn_weights_array)


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='imdb', type=str, help='dataset')
parser.add_argument('--vocab_file', default='wiki.vocab', type=str, help='vocabulary path')
parser.add_argument('--tokenizer', default='sentencepiece', type=str,
                    help='tokenizer to tokenize input corpus. available: sentencepiece, ' + ', '.join(
                        TOKENIZER_CLASSES.keys()))
parser.add_argument('--pretrained_model', default='wiki.model', type=str,
                    help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
parser.add_argument('--output_model_prefix', default='model', type=str, help='output model name prefix')
# Input parameters
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--max_seq_len', default=512, type=int, help='the maximum size of the input sequence')
# Train parameters
parser.add_argument('--epochs', default=30, type=int, help='the number of epochs')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--no_cuda', action='store_true')
# Model parameters
parser.add_argument('--hidden', default=256, type=int, help='the number of expected features in the transformer')
parser.add_argument('--n_layers', default=6, type=int,
                    help='the number of heads in the multi-head attention network')
parser.add_argument('--n_attn_heads', default=8, type=int, help='the number of multi-head attention heads')
parser.add_argument('--dropout', default=0.1, type=float, help='the residual dropout value')
parser.add_argument('--ffn_hidden', default=1024, type=int, help='the dimension of the feedforward network')
parser.add_argument('--save_dir', default='results', type=str, help='save dir')
parser.add_argument("--n_it", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--attn', type=str, default='swd', choices=['trans', 'sink', 'swd'])
parser.add_argument('--visual_attn', type=bool, default=False)
args = parser.parse_args()


save_dir = args.save_dir

try:
    os.mkdir(save_dir)
except:
    pass




main(args)

