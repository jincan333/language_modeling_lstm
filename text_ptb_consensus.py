import hydra
from hydra.utils import instantiate
from upcycle.cuda import try_cuda
from functools import partial
import random
import logging
import traceback
import torch
from torch import nn
import torch.nn.functional as F
import os
import sys
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time


current_script_directory = os.path.dirname(os.path.abspath(__file__))
gnosis_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
sys.path.append(gnosis_directory)

from gnosis import distillation
from gnosis import models as models_class
from gnosis.utils.checkpointing import load_teachers, select_ckpts

from upcycle.scripting import startup
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from gnosis.utils.initialization import interpolate_net

criterion = nn.CrossEntropyLoss()
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous().cuda()
    return data


def get_batch(source, i, seq_len=None):
    seq_len = min(seq_len, source.size(0) - 1)
    if i+seq_len < source.size(0):
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
    else:
        seq_len = source.size(0) - (i+1)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def eval_epoch(nets, data, batch_size, seq_len, epoch, loss_fn, teacher=None, drop_synthetic_inputs=True, with_cka=True):
    for i in range(len(nets)):
        nets[i].eval()
        nets[i].init_hidden(batch_size)
    test_loss = 0
    total = 0
    single_loss = [0 for _ in range(len(nets))]
    perplexity = [0 for _ in range(len(nets))]
    metrics = [{} for _ in range(len(nets))]
    start = time.time()
    with torch.no_grad():
        for k in range(data.size(0)-1):
            # batch = [data[k: k+batch_size], data[k+1: k+1+batch_size]]
            # inputs, targets = try_cuda(*batch[:2])
            inputs, targets = get_batch(data, k, seq_len)
            if drop_synthetic_inputs:
                loss_args = [inputs[:targets.size(0)], targets]  # synthetic data won't be labeled
            else:
                loss_args = [inputs, targets]
            loss, logits = loss_fn(*loss_args)
            test_loss += loss.item()
            total += targets.size(0)
            for i in range(len(nets)):
                single_loss[i] += len(inputs) * F.cross_entropy(logits, targets)
    perplexity = [torch.exp(single_loss[i] / total) for i in range(len(nets))]

    end = time.time()
    for i in range(len(nets)):
        print('[eval] Loss: %.3f | Perplexity_%d: %.3f | Time: %.3f'
                % (test_loss / len(data), i, perplexity[i], end-start))
    print('q: ', F.softmax(loss_fn.q, dim=0))
    
    for i in range(len(nets)):
        metrics[i] = dict(
            test_loss=test_loss / len(data),
            perplexity = perplexity[i],
            epoch=epoch
        )
    return metrics


def consensus_eval_epoch(nets, data, batch_size, seq_len, epoch, loss_fn, teacher=None, drop_synthetic_inputs=True, with_cka=True):
    for i in range(len(nets)):
        nets[i].eval()
        nets[i].init_hidden(batch_size)
    loss_fn.q.requires_grad_(False)
    test_loss = 0
    total = 0
    single_loss = [0 for _ in range(len(nets))]
    perplexity = [0 for _ in range(len(nets))]
    metrics = [{} for _ in range(len(nets))]
    start = time.time()
    with torch.no_grad():
        for k in range(data.size(0)-1):
            # batch = [data[k: k+batch_size], data[k+1: k+1+batch_size]]
            # inputs, targets = try_cuda(*batch[:2])
            inputs, targets = get_batch(data, k, seq_len)
            if drop_synthetic_inputs:
                loss_args = [0, inputs[:targets.size(0)], targets]  # synthetic data won't be labeled
            else:
                loss_args = [0, inputs, targets]
            loss, _, logits_list = loss_fn(*loss_args)
            test_loss += loss.item()
            total += len(inputs)
            for i in range(len(nets)):
                single_loss[i] += len(inputs) * F.cross_entropy(logits_list[i], targets)
    perplexity = [torch.exp(single_loss[i] / total) for i in range(len(nets))]

    end = time.time()
    for i in range(len(nets)):
        print('[eval] Loss: %.3f | Perplexity_%d: %.3f | Time: %.3f'
                % (test_loss / len(data), i, perplexity[i], end-start))
    print('q: ', F.softmax(loss_fn.q, dim=0))
    
    for i in range(len(nets)):
        metrics[i] = dict(
            test_loss=test_loss / len(data),
            perplexity = perplexity[i],
            epoch=epoch
        )
    # only return generalization metrics
    if teacher is None:
        return metrics


def consensus_train_epoch(nets, data, batch_size, seq_len, optimizer, lr_scheduler, epoch, loss_fn):
    print('\nEpoch: %d' % epoch)
    for i in range(len(nets)):
        nets[i].train()
        nets[i].init_hidden(batch_size)
    loss_fn.q.requires_grad_(True)
    train_loss = 0
    total = 0
    single_loss = [0 for _ in range(len(nets))]
    perplexity = [0 for _ in range(len(nets))]
    metrics = [{} for _ in range(len(nets))]
    start = time.time()
    for k in range(data.size(0)-1):
        # batch = [data[k: k+batch_size], data[k+1: k+1+batch_size]]
        # inputs, targets = try_cuda(*batch[:2])
        inputs, targets = get_batch(data, k, seq_len)
        for i in range(len(nets)):
            optimizer.zero_grad()
            loss, _, logits_list = loss_fn(i, inputs, targets)
            loss.backward()
            optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        for i in range(len(nets)):
            single_loss[i] += len(inputs) * F.cross_entropy(logits_list[i], targets)
    perplexity = [torch.exp(single_loss[i] / total) for i in range(len(nets))]
    end = time.time()
    for i in range(len(nets)): 
        print('[LR=%.4f] Loss: %.3f | Perplexity_%d: %.3f%% | Time: %.3f' %
            (lr_scheduler.get_last_lr()[0], train_loss / len(data), i, perplexity[i], end-start))
    print('q: ',F.softmax(loss_fn.q, dim=0))
    lr_scheduler.step()
    for i in range(len(nets)):
        metrics[i] = dict(
                train_loss=train_loss / len(data),
                perplexity=perplexity[i],
                lr=lr_scheduler.get_last_lr()[0],
                epoch=epoch
            )
    return metrics




@hydra.main(config_path='../config', config_name='text_ptb')
def main(config):
    try:
        # construct logger, model, dataloaders
        config, logger = startup(config)
        # config.depth_list=''.join(char for char in str(config.depth_list) if char.isdigit())
        # config.depth_list=[int(config.depth_list[2*i:2*i+2]) for i in range(len(config.depth_list)//2)]
        device = torch.device(f"cuda:{config.gpu}")
        torch.cuda.set_device(int(config.gpu))
        # train_loader, test_loader, train_splits, vocab_size = get_text_loaders(config)
        batch_size = config.dataloader.batch_size
        seq_len = config.dataloader.seq_len
        train_iter = PennTreebank(split='train')
        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        def data_process(raw_text_iter):
            data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
            return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
        
        train_data = data_process(train_iter)
        train_data = batchify(train_data, batch_size)
        test_iter = PennTreebank(split='test')
        test_data = data_process(test_iter)
        test_data = batchify(test_data, batch_size)

        tb_logger = SummaryWriter(log_dir=f"./{config.job_name}/{config.timestamp}")
        tb_prefix="teachers/teacher_/".format(0)
        # if config.teacher.use_ckpts:
        #     try:
        #         teachers, ckpt_files = load_teachers(config, num_words=vocab_size)
        #         if config.teacher.shuffle_ckpts:
        #             print('shuffling checkpoints')
        #             random.shuffle(teachers)
        #     except FileNotFoundError:
        #         teachers = []
        #     if len(teachers) >= config.teacher.num_components:
        #         # use trial_id to determine which checkpoints are used
        #         teachers = select_ckpts(teachers, config.trial_id, config.teacher.num_components, ckpt_names=ckpt_files)
        #         teachers = [try_cuda(m) for m in teachers]
        #         # teachers = [try_cuda(teachers[i]) for i in range(start_idx, stop_idx)]
        # else:
        #     teachers = []
        # num_ckpts = len(teachers)
        criterion = nn.CrossEntropyLoss()
        models=[0 for _ in range(config.models_num)]
        for i in range(config.models_num):
            torch.manual_seed(i)
            # config.classifier.depth = config.depth_list[i]
            # print('depth:', config.classifier.depth)
            models[i]=hydra.utils.instantiate(config.classifier, vocab_size=len(vocab))
            print(f'model{i}, vocab{len(vocab)}')
            models[i]=try_cuda(models[i])
            models[i].init_weights()
            logger.save_obj(models[i].state_dict(), f'teacher_init_{i}.ckpt')
            if config.loss.name == 'base':
                consensus_loss = distillation.ClassifierTeacherLoss(models[i])

        print(f"==== training consensus model ====")
        if config.loss.name == 'consensus_all':
            consensus_loss = distillation.ClassifierConsensusAllLoss(models, config)
        elif config.loss.name == 'consensus':
            consensus_loss = distillation.ClassifierConsensusLoss(models, config)
        elif config.loss.name == 'consensus_detach':
            consensus_loss = distillation.ClassifierConsensusLoss_detach(models, config)
        elif config.loss.name == 'consensus_exclude':
            consensus_loss = distillation.ClassifierConsensusExcludeLoss(models, config)
        elif config.loss.name == 'consensus_exclude_ptb':
            consensus_loss = distillation.ClassifierConsensusExcludeLossPTB(models, config)
        elif config.loss.name == 'consensus_together':
            consensus_loss = distillation.ClassifierConsensusTogetherLoss(models, config)
        elif config.loss.name == 'consensus_together_direct':
            consensus_loss = distillation.ClassifierConsensusTogetherDirectLoss(models, config)
        else:
            raise ("loss does not exist")
        
        records = [[] for _ in range(config.models_num)] 
        params = []
        for model in models:
            for param in model.parameters():
                params.append(param)
        params.append(consensus_loss.q)
        optimizer = instantiate(config.trainer.optimizer, params=params)
        lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)
        eval_metrics = consensus_eval_epoch(models, test_data, batch_size, seq_len, epoch=0, loss_fn=consensus_loss)
        for i in range(config.models_num):
            records[i].append(eval_metrics[i])
        for epoch in range(config.trainer.num_epochs):
            metrics = [{} for _ in range(config.models_num)]
            train_metrics = consensus_train_epoch(models, train_data, batch_size, seq_len, optimizer, lr_scheduler, epoch=epoch+1, loss_fn=consensus_loss)
            for i in range(config.models_num):
                metrics[i].update(train_metrics[i])
            if epoch % config.trainer.eval_period < (config.trainer.eval_period - 1):
                continue
            eval_metrics = consensus_eval_epoch(models, test_data, batch_size, seq_len, epoch=epoch + 1, loss_fn=consensus_loss)
            for i in range(config.models_num):
                metrics[i].update(eval_metrics[i])
                records[i].append(metrics[i])
                # log to tensorboard
                for key, val in train_metrics[i].items():
                    if key == 'epoch':
                        continue
                    tb_logger.add_scalar(f"{tb_prefix}train/{key}", val, epoch)
                for key, val in eval_metrics[i].items():
                    if key == 'epoch':
                        continue
                    tb_logger.add_scalar(f"{tb_prefix}eval/{key}", val, epoch)

        for i in range(config.models_num):
            logger.add_table(f'teacher_{i}_train_metrics', records[i])
            logger.write_csv()
            logger.save_obj(models[i].state_dict(), f'teacher_{i}.ckpt')

        print('==== ensembling classifiers ====')
        teacher = [models_class.ClassifierEnsemble(*models)]
        teacher_test_metrics = eval_epoch(teacher, test_data, batch_size, seq_len, epoch=0,
                                    loss_fn=models_class.ensemble.ClassifierEnsembleLoss(teacher[0]))

        return



        # distill_splits = [train_splits[i] for i in config.distill_loader.splits]
        distill_loader = hydra.utils.instantiate(config.distill_loader, loader=train_loader, teacher=teacher)
        teacher_train_metrics = eval_epoch(teacher, distill_loader, epoch=0,
                                           loss_fn=models_class.ensemble.ClassifierEnsembleLoss(teacher),
                                           drop_synthetic_inputs=False)

        student = hydra.utils.instantiate(config.classifier, num_words=vocab_size)
        student = try_cuda(student)

        if config.teacher.ckpt_init.type == 'init':
            assert config.classifier.depth == config.teacher.depth
            assert config.teacher.num_components == 1
            init_teachers, init_fnames = load_teachers(config, ckpt_pattern='*teacher_init_?.ckpt',
                                                       num_words=vocab_size)
            print('initializing the student near the initial teacher weights')
            init_teachers = select_ckpts(init_teachers, config.trial_id, 1, ckpt_names=init_fnames)
            start_idx = (config.trial_id * config.teacher.num_components) % len(init_teachers) - len(init_teachers)
            stop_idx = start_idx + 1
            print(f'using checkpoints {[(len(init_teachers) + i) % len(init_teachers) for i in range(start_idx, stop_idx)]}')
            student = interpolate_net(student, init_teachers[0].state_dict(),
                                      config.teacher.ckpt_init.loc_param, train_loader,
                                      config.trainer.freeze_bn)

        elif config.teacher.ckpt_init.type == 'final':
            assert config.classifier.depth == config.teacher.depth
            assert config.teacher.num_components == 1
            print('initializing the student near the final teacher weights')
            student = interpolate_net(student, teachers[0].state_dict(),
                                      config.teacher.ckpt_init.loc_param, train_loader,
                                      config.trainer.freeze_bn)
            # scale the learning rate down if student is initialized close to teacher
            config.trainer.optimizer.lr = max(
                config.trainer.optimizer.lr * config.teacher.ckpt_init.loc_param,
                config.trainer.lr_scheduler.eta_min
            )
        logger.save_obj(student.state_dict(), 'student_init.ckpt')

        # train_loader, synth_loader = get_distill_loaders(config, train_loader, None)
        student_base_loss = hydra.utils.instantiate(config.loss.init)
        student_loss = distillation.ClassifierStudentLoss(student, student_base_loss, config.loss.alpha)

        print(f"==== distilling student classifier ====")
        student, records = train_loop(
            config,
            student,
            train_closure=distillation_epoch,
            train_loader=distill_loader,
            train_kwargs=dict(loss_fn=student_loss),
            eval_closure=eval_epoch,
            eval_loader=test_loader,
            eval_kwargs=dict(loss_fn=student_loss, teacher=teacher,
                             drop_synthetic_inputs=False, with_cka=False),
            tb_logger=tb_logger,
            tb_prefix="student/",
        )
        for r in records:
            r.update(dict(teacher_train_acc=teacher_train_metrics['test_acc'],
                          teacher_test_acc=teacher_test_metrics['test_acc']))
        logger.add_table(f'student_train_metrics', records)
        logger.write_csv()
        logger.save_obj(student.state_dict(), f'student.ckpt')

        del train_loader, test_loader  # these will be regenerated w/o augmentation
        # save_logits(config, student, teacher, None, logger)

        return 1 - records[-1]['test_acc'] / 100. if len(records) > 0 else float('NaN')

    except Exception:
        logging.error(traceback.format_exc())
        return float('NaN')


if __name__ == '__main__':
    main()
