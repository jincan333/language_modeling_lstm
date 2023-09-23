import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import json
import data
import model
import torch.nn.functional as F

from utils import batchify, get_batch, repackage_hidden
from classification import ClassifierConsensusExcludeLossPTB, kl_div_softmax, kl_div_logits

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
# consensus
parser.add_argument('--exp_name', type=str, default='lstm_ptb')
parser.add_argument('--loss', type=str, default='consensus_exclude')
parser.add_argument('--consensus_alpha', type=float, default=1)
parser.add_argument('--models_num', type=int, default=1)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--learnable_q', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)

# original
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=3,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.40,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=141,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', default=True,help='use CUDA')
parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='ckpt/consensus'+randomhash+'PTB.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
torch.cuda.set_device(int(args.gpu))
###############################################################################
# Load data
###############################################################################

def model_save(fn, k):
    fn=fn+'_'+str(k)
    with open(fn, 'wb') as f:
        torch.save([models[k], criterion, optimizer], f)

def model_load(fn, k):
    fn=fn+'_'+str(k)
    global models, criterion, optimizer
    with open(fn, 'rb') as f:
        models[k], criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
models=[0 for _ in range(args.models_num)]
for k in range(args.models_num):
    torch.manual_seed(k)
    models[k]=model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    print(f'model{k}')
    models[k]=models[k].cuda()
    models[k].init_weights()
consensus_loss = ClassifierConsensusExcludeLossPTB(models, args)

if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False).cuda()

###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###

###
# if args.cuda:
#     model = model.cuda()
#     criterion = criterion.cuda()
###
params = []
models_params = []
for k in range(args.models_num):
    models_params += list(models[k].parameters())
    params += list(models[k].parameters())
params+=list(criterion.parameters())
params.append(consensus_loss.q)
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args: ',json.dumps(vars(args), indent=4))
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def consensus_evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    for k in range(args.models_num):
        models[k].eval()
    consensus_loss.q.requires_grad_(False)
    if args.model == 'QRNN': model.reset()
    ppls = [0 for _ in range(args.models_num)]
    for k in range(args.models_num):
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = models[k].init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            output, hidden = models[k](data, hidden)
            total_loss += len(data) * criterion(models[k].decoder.weight, models[k].decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        ppls[k] = total_loss.item() / len(data_source)
    return ppls

def consensus_train():
    for k in range(args.models_num):
        models[k].train()
    consensus_loss.q.requires_grad_(True)
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_losses = [0 for _ in range(args.models_num)]
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hiddens = [models[k].init_hidden(args.batch_size) for k in range(args.models_num)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        for circle in range(args.models_num):
            hiddens = [repackage_hidden(hiddens[k]) for k in range(args.models_num)]
            optimizer.zero_grad()
            raw_losses=[0 for _ in range(args.models_num)]
            loss=0
            outputs=[0 for _ in range(args.models_num)]
            rnn_hss=[0 for _ in range(args.models_num)]
            dropped_rnn_hss=[0 for _ in range(args.models_num)]
            for k in range(args.models_num):
                outputs[k], hiddens[k], rnn_hss[k], dropped_rnn_hss[k] = models[k](data, hiddens[k], return_h=True)
                raw_loss = criterion(models[k].decoder.weight, models[k].decoder.bias, outputs[k], targets)
                loss+=raw_loss  
                raw_losses[k]=raw_loss
                # Activiation Regularization
                if args.alpha: loss += sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hss[k][-1:])
                # Temporal Activation Regularization (slowness)
                if args.beta: loss += sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hss[k][-1:])
                
            q_softmaxes = [0 for _ in range(args.models_num)]
            kl_loss = 0
            for k in range(args.models_num):
                q_masked = consensus_loss.q*consensus_loss.mask[k] - (1 - consensus_loss.mask[k])*1e10
                q_softmaxes[k] = F.softmax(q_masked, dim=0) if consensus_loss.learnable_q else F.softmax(torch.zeros(consensus_loss.models_num), dim=0)
                for l in range(args.models_num):
                    if consensus_loss.detach:
                        kl_loss += q_softmaxes[k][l] * (kl_div_logits(outputs[k], outputs[l].detach(), consensus_loss.T) + 
                                                kl_div_logits(outputs[l].detach(), outputs[k], consensus_loss.T))
                    else:
                        kl_loss += q_softmaxes[k][l] * (kl_div_logits(outputs[k], outputs[l], consensus_loss.T) + 
                                                kl_div_logits(outputs[l], outputs[k], consensus_loss.T))
            loss += args.consensus_alpha*kl_loss        
            
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()

        total_losses =[total_losses[k]+ raw_losses[k].data for k in range(args.models_num)]
        optimizer.param_groups[0]['lr'] = lr2
        cur_losses=[0 for _ in range(args.models_num)]
        if batch % args.log_interval == 0 and batch > 0:
            for k in range(args.models_num):
                cur_losses[k] = total_losses[k].item() / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f} | model {:1d}'.format(
                    epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_losses[k], math.exp(cur_losses[k]), cur_losses[k] / math.log(2), k))
                total_losses[k] = 0
                start_time = time.time()
            print('q:', q_softmaxes)
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_losses = [[] for _ in range(args.models_num)]
stored_losses = [100000000 for _ in range(args.models_num)]

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        consensus_train()
        if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for k in range(args.models_num):
                    for name,prm in models[k].named_parameters():
                        tmp[name+'_'+str(k)] = prm.data.clone()
                        if 'ax' in optimizer.state[prm]:
                            prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2es = consensus_evaluate(val_data)

                for k in range(args.models_num):
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f} | model {:1d} | '.format(
                            epoch, (time.time() - epoch_start_time), val_loss2es[k], math.exp(val_loss2es[k]), val_loss2es[k] / math.log(2), k))
                    print('-' * 89)

                    if val_loss2es[k] < stored_losses[k]:
                        model_save(args.save, k)
                        print('Saving Averaged!')
                        stored_losses[k] = val_loss2es[k]
                for k in range(args.models_num):
                    for name,prm in models[k].named_parameters():
                        prm.data = tmp[name+'_'+str(k)].clone()

        else:
            val_losses = consensus_evaluate(val_data, eval_batch_size)
            for k in range(args.models_num):
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f} | model {:1d}'.format(
                epoch, (time.time() - epoch_start_time), val_losses[k], math.exp(val_losses[k]), val_losses[k] / math.log(2), k))
                print('-' * 89)
                if val_losses[k] < stored_losses[k]:
                    model_save(args.save, k)
                    print(f'Saving model {k} (new best validation)')
                    stored_losses[k] = val_losses[k]

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_losses[k])>args.nonmono and val_losses[k] > min(best_val_losses[k][:-args.nonmono])):
                # if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0]:
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(models_params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch), k)
                    print('Dividing learning rate by 10')
                    # TODO
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_losses[k].append(val_losses[k])

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
for k in range(args.models_num):
    model_load(args.save, k)

# Run on test data.
test_losses = consensus_evaluate(test_data, test_batch_size)
for k in range(args.models_num):
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_losses[k], math.exp(test_losses[k]), test_losses[k] / math.log(2)))
    print('=' * 89)
