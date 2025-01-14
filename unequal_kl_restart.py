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
parser.add_argument('--loss', type=str, default='consensus_fifth')
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--models_num', type=int, default=2)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--learnable_q', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0,help='random seed')
parser.add_argument('--T', type=float, default=1.5)
parser.add_argument('--momentum', type=float, default=0.3)
parser.add_argument('--student_lr', type=float, default=30)
parser.add_argument('--lr_gamma', type=float, default=0.25, help='forth best:0.25, ')
parser.add_argument('--student_steps', type=int, default=1, help='')
parser.add_argument('--current_index', type=int, default=0, help='')
parser.add_argument('--student_epochs', type=int, default=5)
parser.add_argument('--distill_epochs', type=int, default=5)
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
parser.add_argument('--decreasing_step', type=list, default=[0.4, 0.65, 0.75, 0.83])
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='ckpt/baseline'+randomhash+'PTB.pt',
                    help='path to save the final model')
parser.add_argument('--opt', type=str,  default='SGD',
                    help='SGD, Adam, RMSprop, Momentum')
args = parser.parse_args()
print(json.dumps(vars(args), indent=4))
wandb.login(key='ca2f2a2ae6e84e31bbc09a8f35f9b9a534dfbe9b')
wandb.init(project='ensemble_distill_unequal_kl_restart', entity='jincan333', name=args.exp_name)
torch.cuda.set_device(int(args.gpu))
def set_random_seed(s):
    np.random.seed(args.seed+s)
    torch.manual_seed(args.seed+s)
    torch.cuda.manual_seed(args.seed+s)


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


set_random_seed(0)
corpus = corpus.Corpus(args.data)
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
    set_random_seed(k)
    models[k] = model.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).cuda()
    print(models[k])
    total_params += sum(p.numel() for p in models[k].parameters())
    # print(f'Current parameters: {total_params}')
print(f"Total parameters: {total_params}")

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
    print('q:', consensus_loss.q)
    return losses


def train():
    # choose a optimizer
    for k in range(args.models_num):
        models[k].train()
    consensus_loss.q.requires_grad_(True)
    total_loss = 0
    start_time = time.time()
    teacher_hiddenes = [models[k].init_hidden(args.batch_size) for k in range(args.models_num)]
    student_hiddenes = [models[k].init_hidden(args.batch_size) for k in range(args.models_num)]
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # update teacher
        teacher_hiddenes = [repackage_hidden(teacher_hiddenes[l]) for l in range(args.models_num)]
        logits_list = [0 for _ in range(args.models_num)]
        for k in range(args.models_num):
            logits_list[k], teacher_hiddenes[k] = models[k](data, teacher_hiddenes[k])
        if args.detach:
            teacher_loss=F.cross_entropy(logits_list[0], targets)+ \
                        args.alpha*(kl_div_logits(logits_list[0], logits_list[1].detach(), args.T))
        else:
            teacher_loss=F.cross_entropy(logits_list[0], targets)+ \
                        args.alpha*(kl_div_logits(logits_list[0], logits_list[1], args.T))
        opt.zero_grad()
        teacher_loss.backward()
        torch.nn.utils.clip_grad_norm_(models[0].parameters(), args.clip)
        opt.step()
        
        # update student
        for _ in range(args.student_steps):
            data, targets = get_batch(train_data, args.current_index)
            args.current_index = (args.bptt+args.current_index)%(train_data.size(0) - 1)
            student_hiddenes = [repackage_hidden(student_hiddenes[l]) for l in range(args.models_num)]
            for k in range(args.models_num):
                logits_list[k], student_hiddenes[k] = models[k](data, student_hiddenes[k])
            if args.detach:
                student_loss=kl_div_logits(logits_list[1], logits_list[0].detach(), args.T)
            else:
                student_loss=kl_div_logits(logits_list[1], logits_list[0], args.T)
            student_opt.zero_grad()
            student_loss.backward()
            torch.nn.utils.clip_grad_norm_(models[1].parameters(), args.clip)
            student_opt.step()

        total_loss += teacher_loss

    cur_loss = total_loss / interval
    elapsed = time.time() - start_time
    print('| epoch {:3d} | {:5d}/{:5d} batches | teacher lr {:02.2f} | student lr {:02.2f} | ms/batch {:5.2f} | '
            'loss {:5.2f}'.format(
        epoch, batch, len(train_data) // args.bptt, opt.param_groups[0]['lr'], student_opt.param_groups[0]['lr'],
        elapsed * 1000 / interval, cur_loss))
    wandb.log({'teacher lr': opt.param_groups[0]['lr'], 'student lr': student_opt.param_groups[0]['lr']}, step=epoch+(epoch // args.distill_epochs)*args.student_epochs)
    total_loss = 0
    start_time = time.time()


def student_retrain():
    for e in range(args.student_epochs):
        models[1].train()
        total_loss = 0
        start_time = time.time()
        hiddenes = [models[k].init_hidden(args.batch_size) for k in range(args.models_num)]
        # train_data size(batchcnt, bsz)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, i)
            hiddenes = [repackage_hidden(hiddenes[l]) for l in range(args.models_num)]
            logits_list = [0 for _ in range(args.models_num)]
            for k in range(args.models_num):
                logits_list[k], hiddenes[k] = models[k](data, hiddenes[k])
            if args.detach:
                loss=kl_div_logits(logits_list[1], logits_list[0].detach(), args.T)
            else:
                loss=kl_div_logits(logits_list[1], logits_list[0], args.T)
            student_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[1].parameters(), args.clip)
            student_opt.step()
            total_loss += loss.data
        
        cur_loss = total_loss / len(train_data)
        elapsed = time.time() - start_time
        print('|student epoch {:3d} | {:5d}/{:5d} batches | student lr {:02.2f} | time {:5.2f} | '
                'train loss {:5.2f}'.format(
            epoch, batch, len(train_data) // args.bptt, student_opt.param_groups[0]['lr'], elapsed, cur_loss))
        
        val_losses = evaluate(val_data)
        for k in range(args.models_num):
            print('|student epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - start_time), val_losses[k], math.exp(val_losses[k])))
        if best_val_losses[1] < val_losses[1] and e!=0:
            student_lr = student_opt.param_groups[0]['lr']
            student_lr *= args.lr_gamma
            for group in student_opt.param_groups:
                group['lr'] = student_lr
        if val_losses[1] < best_val_losses[1] or e == 0:
            best_val_losses[1] = val_losses[1]

        # student_scheduler.step()
        wandb.log({'student lr': student_opt.param_groups[0]['lr'], 'student valid ppl': math.exp(val_losses[1])}, step=epoch+e+(epoch // args.distill_epochs - 1)*args.student_epochs)
        total_loss = 0
        start_time = time.time()


# Loop over epochs.
best_val_losses = [None for _ in range(args.models_num)]

opt = torch.optim.SGD(models[0].parameters(), lr=args.lr, momentum=args.momentum)
student_opt= torch.optim.SGD(models[1].parameters(), lr=args.student_lr, momentum=args.momentum)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.epochs * _) for _ in args.decreasing_step], gamma=args.lr_gamma)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

# student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_opt, milestones=[int(args.epochs * _) for _ in args.decreasing_step], gamma=args.lr_gamma)
# student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_opt, T_max=(args.student_epochs+args.distill_epochs))

try:
    for epoch in range(1, args.epochs+1):
        if epoch % args.distill_epochs == 0 and epoch!=args.epochs:
            set_random_seed(epoch)
            models[1] = model.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).cuda()
            student_opt=torch.optim.SGD(models[1].parameters(), lr=args.student_lr, momentum=args.momentum)
            # student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_opt, T_max=(args.student_epochs+args.distill_epochs))
            student_retrain()
            # student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_opt, milestones=[int(args.epochs * _) - args.student_epochs for _ in args.decreasing_step], gamma=args.lr_gamma)
        epoch_start_time = time.time()
        train()
        val_losses = evaluate(val_data)
        thres=0
        scheduler.step()
        # student_scheduler.step()
        if best_val_losses[0] and sum([math.exp(_) for _ in best_val_losses]) < sum([math.exp(_) for _ in val_losses]) and (epoch % args.distill_epochs != 0 or epoch==args.epochs):
            # if best_val_losses[0] < val_losses[0]:
            #     lr = opt.param_groups[0]['lr']
            #     lr *= args.lr_gamma
            #     for group in opt.param_groups:
            #         group['lr'] = lr
            if best_val_losses[1] < val_losses[1]:
                student_lr = student_opt.param_groups[0]['lr']
                student_lr *= args.lr_gamma
                for group in student_opt.param_groups:
                    group['lr'] = student_lr
        for k in range(args.models_num):
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_losses[k], math.exp(val_losses[k])))
            print('-' * 89)
            if not best_val_losses[k] or val_losses[k] < best_val_losses[k] or (epoch % args.distill_epochs == 0 and epoch!=args.epochs):
                with open(args.save+'_'+str(k), 'wb') as f:
                    torch.save(models[k], f)
                best_val_losses[k] = val_losses[k]
        wandb.log({'teacher valid ppl': math.exp(val_losses[0])}, step=epoch+(epoch // args.distill_epochs)*args.student_epochs)
        wandb.log({'student valid ppl': math.exp(val_losses[1])}, step=epoch+(epoch // args.distill_epochs)*args.student_epochs)

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
wandb.log({'teacher test ppl': math.exp(test_losses[0])}, step=epoch+(epoch // args.distill_epochs)*args.student_epochs)
wandb.log({'student test ppl': math.exp(test_losses[1])}, step=epoch+(epoch // args.distill_epochs)*args.student_epochs)
wandb.finish()
