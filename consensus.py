import argparse
import time
import math
import torch
import torch.nn as nn
import corpus
import model
import numpy as np
import torch.nn.functional as F
from classification import ClassifierConsensusExcludeLossPTB, ClassifierConsensusForthLossPTB, ClassifierConsensusFifthLossPTB, ClassifierConsensusForthSymmetrizedLossPTB, ClassifierConsensusFifthSymmetrizedLossPTB
import json
import wandb

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

# consensus
parser.add_argument('--exp_name', type=str, default='lstm_ptb')
parser.add_argument('--loss', type=str, default='consensus_forth')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--models_num', type=int, default=2)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--learnable_q', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0,help='random seed')
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--momentum', type=float, default=0.3)
# original
parser.add_argument('--data', type=str, default='./input', # /input
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--emsize', type=int, default=650)
parser.add_argument('--nhid', type=int, default=650)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--lr', type=float, default=30)
parser.add_argument('--clip', type=float, default=0.20)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--bptt', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.45)
parser.add_argument('--decreasing_step', type=list, default=[0.6, 0.75, 0.9])
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='ckpt/baseline'+randomhash+'PTB.pt',
                    help='path to save the final model')
parser.add_argument('--opt', type=str,  default='SGD',
                    help='SGD, Adam, RMSprop, Momentum')
args = parser.parse_args()
print(json.dumps(vars(args), indent=4))
wandb.login(key='ca2f2a2ae6e84e31bbc09a8f35f9b9a534dfbe9b')
wandb.init(project='ensemble_distill_consensus', entity='jincan333', name=args.exp_name)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.set_device(int(args.gpu))
args.clip=args.clip*math.sqrt(args.models_num)


# Load data
corpus = corpus.Corpus(args.data)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size) # size(total_len//bsz, bsz)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Build the model
interval = 200 # interval to report
ntokens = len(corpus.dictionary) # 10000

models = [0 for _ in range(args.models_num)]
total_params = 0
for k in range(args.models_num):
    np.random.seed(k+args.seed)
    torch.manual_seed(k+args.seed)
    torch.cuda.manual_seed(k+args.seed)
    models[k] = model.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).cuda()
    print(models[k])
    total_params += sum(p.numel() for p in models[k].parameters())
    # print(f'Current parameters: {total_params}')
print(f"Total parameters: {total_params}")

# Load checkpoint
# if args.checkpoint != '':
#     model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage).cuda()


criterion = nn.CrossEntropyLoss().cuda()
if args.loss=='consensus_exclude':
    consensus_loss = ClassifierConsensusExcludeLossPTB(models, args)
elif args.loss=='consensus_forth':
    consensus_loss = ClassifierConsensusForthLossPTB(models, args)
elif args.loss=='consensus_fifth':
    consensus_loss = ClassifierConsensusFifthLossPTB(models, args)
elif args.loss=='consensus_forth_symmetrized':
    consensus_loss = ClassifierConsensusForthSymmetrizedLossPTB(models, args)
elif args.loss=='consensus_fifth_symmetrized':
    consensus_loss = ClassifierConsensusFifthSymmetrizedLossPTB(models, args)
# Training code

def repackage_hidden(h):
    # detach
    return tuple(v.clone().detach() for v in h)


def get_batch(source, i):
    # source: size(total_len//bsz, bsz)
    seq_len = min(args.bptt, len(source) - 1 - i)
    #data = torch.tensor(source[i:i+seq_len]) # size(bptt, bsz)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    #target = torch.tensor(source[i+1:i+1+seq_len].view(-1)) # size(bptt * bsz)
    return data.cuda(), target.cuda()


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    losses = [0 for _ in range(args.models_num)]
    consensus_loss.q.requires_grad_(False)
    for k in range(args.models_num):
        with torch.no_grad():
            models[k].eval()
            total_loss = 0
            ntokens = len(corpus.dictionary)
            hidden = models[k].init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
            for i in range(0, data_source.size(0) - 1, args.bptt):# iterate over every timestep
                data, targets = get_batch(data_source, i)
                output, hidden = models[k](data, hidden)
                # model input and output
                # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
                # output size(bptt*bsz, ntoken)
                total_loss += len(data) * criterion(output, targets).data
                hidden = repackage_hidden(hidden)
        losses[k] = total_loss / len(data_source)
    # q_softmax = F.softmax(consensus_loss.q, dim=0) if consensus_loss.learnable_q else F.softmax(torch.zeros(consensus_loss.models_num), dim=0)
    print('q:', consensus_loss.q)
    return losses


def train():
    # choose a optimizer
    for k in range(args.models_num):
        models[k].train()
    consensus_loss.q.requires_grad_(True)
    total_loss = 0
    start_time = time.time()
    hiddenes = [models[k].init_hidden(args.batch_size) for k in range(args.models_num)]
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # for k in range(args.models_num):
        hiddenes = [repackage_hidden(hiddenes[l]) for l in range(args.models_num)]
        loss, _, logits, hiddenes = consensus_loss(0, data, hiddenes, targets)
        opt.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()

        total_loss += loss

    cur_loss = total_loss / interval
    elapsed = time.time() - start_time
    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            'loss {:5.2f}'.format(
        epoch, batch, len(train_data) // args.bptt, opt.param_groups[0]['lr'],
        elapsed * 1000 / interval, cur_loss))
    wandb.log({'lr': opt.param_groups[0]['lr'], 'train loss': cur_loss, 'batches': batch, 'train time(s)': elapsed}, step=epoch)
    total_loss = 0
    start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_losses = [None for _ in range(args.models_num)]

params = []
for k in range(args.models_num):
    params += list(models[k].parameters())
params.append(consensus_loss.q)

opt = torch.optim.SGD(params, lr=lr, momentum=args.momentum)
if args.opt == 'Adam':
    opt = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.99))
    lr = 0.001
if args.opt == 'Momentum':
    opt = torch.optim.SGD(params, lr=lr, momentum=0.8)
if args.opt == 'RMSprop':
    opt = torch.optim.RMSprop(params, lr=0.001, alpha=0.9)
    lr = 0.001
# scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.epochs * _) for _ in args.decreasing_step], gamma=0.25)

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_losses = evaluate(val_data)
        thres=0
        # scheduler.step()
        if best_val_losses[0] and sum([math.exp(_) for _ in best_val_losses]) < sum([math.exp(_) for _ in val_losses]):
        # if sum(best_val_losses) < sum(val_losses):
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if args.opt == 'SGD' or args.opt == 'Momentum':
                lr /= 4.0
                for group in opt.param_groups:
                    group['lr'] = lr
        for k in range(args.models_num):
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_losses[k], math.exp(val_losses[k])))
            print('-' * 89)
            wandb.log({'valid loss '+str(k): val_losses[k], 'valid ppl '+str(k): math.exp(val_losses[k])}, step=epoch)
        # Save the model if the validation loss is the best we've seen so far.
            if not best_val_losses[k] or val_losses[k] < best_val_losses[k]:
                with open(args.save+'_'+str(k), 'wb') as f:
                    torch.save(models[k], f)
                best_val_losses[k] = val_losses[k]
        wandb.log({'epoch time': time.time() - epoch_start_time}, step=epoch)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
for k in range(args.models_num):
    with open(args.save+'_'+str(k), 'rb') as f:
        models[k] = torch.load(f)

# Run on test data.
test_losses = evaluate(test_data)
for k in range(args.models_num):
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_losses[k], math.exp(test_losses[k])))
    print('=' * 89)
    wandb.log({'test loss '+str(k): test_losses[k], 'test ppl '+str(k): math.exp(test_losses[k])}, step=epoch)
wandb.finish()
