import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import kl_divergence
import math
from abc import ABC, abstractmethod
import numpy as np


def kl_div_softmax(p, q, T):
    loss = F.kl_div(F.log_softmax(p/T, dim=1), F.softmax(q/T, dim=1), reduction='batchmean') * T * T
    return loss


def kl_div_logits(p, q, T):
    loss_func = nn.KLDivLoss(reduction = 'batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


def kl_divergence_loss():
    pass


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1), F.softmax(out_t/self.T, dim=1), reduction='batchmean') * self.T * self.T
        return loss


class ClassifierTeacherLoss(object):
    def __init__(self, teacher_model):
        self.teacher = teacher_model

    def __call__(self, inputs, targets):
        logits = self.teacher(inputs)
        loss = F.cross_entropy(logits, targets)
        return loss, logits


class ClassifierTeacherPlusLoss(object):
    def __init__(self, teacher_model1, teacher_model2, alpha):
        self.teacher1 = teacher_model1
        self.teacher2 = teacher_model2
        self.alpha = alpha
        # self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def __call__(self, inputs, targets):
        logits1 = self.teacher1(inputs)
        logits2 = self.teacher2(inputs)
        teacher1_dist = Categorical(logits=logits1)
        teacher2_dist = Categorical(logits=logits2)

        kl_p_q = kl_divergence(teacher1_dist, teacher2_dist).mean()
        kl_q_p = kl_divergence(teacher2_dist, teacher1_dist).mean()
        kl_loss =  kl_p_q + kl_q_p
        ce_loss = F.cross_entropy(logits1, targets) + F.cross_entropy(logits2, targets)
        alpha=self.alpha
        # alpha = torch.sigmoid(self.alpha)
        loss = alpha*kl_loss + (1 - alpha)*ce_loss
        return loss, logits1, logits2


class ClassifierMultiTeacherLoss(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = config.loss.alpha
        # self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, inputs, targets):
        logits_list = [0 for _ in range(self.models_num)]
        models_dists = [0 for _ in range(self.models_num)]
        ce_loss = 0
        for i in range(self.models_num):
            logits_list[i] = self.models[i](inputs)
            models_dists[i] = Categorical(logits = logits_list[i])
            ce_loss += F.cross_entropy(logits_list[i], targets)
        kl_loss = 0
        for i in range(self.models_num):
            for j in range(i+1, self.models_num):
                kl_loss += kl_divergence(models_dists[i], models_dists[j]).mean() + kl_divergence(models_dists[j], models_dists[i]).mean()
        alpha=self.alpha
        # alpha = torch.sigmoid(self.alpha)
        loss = alpha*kl_loss + ce_loss
        return loss, logits_list
    

class ClassifierConsensusLoss(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = 2 / self.models_num if config.loss.alpha == -1 else config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        # self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        logits_list = [0 for _ in range(self.models_num)]
        models_dists = [0 for _ in range(self.models_num)]
        net_logits = self.models[index](inputs)
        net_dists = Categorical(logits = net_logits)
        ce_loss = F.cross_entropy(net_logits, targets)
        for i in range(self.models_num):
            if i == index:
                logits_list[i] = net_logits
            else:
                logits_list[i] = self.models[i](inputs)
            # logits_list[i] = self.models[i](inputs).detach()
            models_dists[i] = Categorical(logits = logits_list[i])
        kl_loss = 0
        q_softmax = F.softmax(self.q, dim=0)
        for i in range(self.models_num):
            # if i == index:
            #     continue
            # kl_loss += q_softmax[i] * (kl_divergence(models_dists[i], net_dists).mean() + kl_divergence(net_dists, models_dists[i]).mean())
            kl_loss += (kl_divergence(models_dists[i], net_dists).mean() + kl_divergence(net_dists, models_dists[i]).mean())
        # alpha = torch.sigmoid(self.alpha)
        loss = self.alpha*kl_loss + ce_loss
        return loss, net_logits, logits_list


class ClassifierConsensusLoss_detach(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = 2 / self.models_num if config.loss.alpha == -1 else config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        # self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        logits_list = [0 for _ in range(self.models_num)]
        models_dists = [0 for _ in range(self.models_num)]
        net_logits = self.models[index](inputs)
        net_dists = Categorical(logits = net_logits)
        ce_loss = F.cross_entropy(net_logits, targets)
        for i in range(self.models_num):
            if i == index:
                logits_list[i] = net_logits
            else:
                logits_list[i] = self.models[i](inputs).detach()
            models_dists[i] = Categorical(logits = logits_list[i])
        kl_loss = 0
        q_softmax = F.softmax(self.q, dim=0)
        for i in range(self.models_num):
            # if i == index:
            #     continue
            kl_loss += q_softmax[i] * (kl_divergence(models_dists[i], net_dists).mean() + kl_divergence(net_dists, models_dists[i]).mean())
        # alpha = torch.sigmoid(self.alpha)
        loss = self.alpha*kl_loss + ce_loss
        return loss, net_logits, logits_list



class ClassifierConsensusAllLoss(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        # self.alpha = torch.nn.Parameter(torch.tensor(alpha))
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        logits_list = [0 for _ in range(self.models_num)]
        ce_loss=0
        for i in range(self.models_num):
            logits_list[i] = self.models[i](inputs)
            ce_loss += F.cross_entropy(logits_list[i], targets)
        kl_loss = 0
        q_softmax = F.softmax(self.q, dim=0) if self.learnable_q else F.softmax(torch.zeros(self.models_num), dim=0)
        for i in range(self.models_num):
            for j in range(self.models_num):
                if self.detach:
                    kl_loss += q_softmax[i] * (kl_div_softmax(logits_list[i], logits_list[j].detach(), self.T) + 
                                               kl_div_softmax(logits_list[j].detach(), logits_list[i], self.T))
                else:
                    kl_loss += q_softmax[i] * (kl_div_softmax(logits_list[i], logits_list[j], self.T) + 
                                               kl_div_softmax(logits_list[j], logits_list[i], self.T))
        loss = self.alpha*kl_loss + ce_loss
        return loss, logits_list[index], logits_list


class ClassifierConsensusExcludeLoss(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        self.mask = torch.ones((config.models_num, config.models_num), requires_grad=False)
        for i in range(self.models_num):
            self.mask[i, i] = 0
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        logits_list = [0 for _ in range(self.models_num)]
        q_softmaxes = [0 for _ in range(self.models_num)]
        ce_loss=0
        kl_loss = 0
        for i in range(self.models_num):
            logits_list[i] = self.models[i](inputs)
            ce_loss += F.cross_entropy(logits_list[i], targets)
        for i in range(self.models_num):
            q_masked = self.q*self.mask[i] - (1 - self.mask[i])*1e10
            q_softmaxes[i] = F.softmax(q_masked, dim=0) if self.learnable_q else F.softmax(torch.zeros(self.models_num), dim=0)
            for j in range(self.models_num):
                if self.detach:
                    kl_loss += q_softmaxes[i][j] * (kl_div_logits(logits_list[i], logits_list[j].detach(), self.T) + 
                                               kl_div_logits(logits_list[j].detach(), logits_list[i], self.T))
                else:
                    kl_loss += q_softmaxes[i][j] * (kl_div_logits(logits_list[i], logits_list[j], self.T) + 
                                               kl_div_logits(logits_list[j], logits_list[i], self.T))
        loss = self.alpha*kl_loss + ce_loss
        return loss, logits_list[index], logits_list



class ClassifierConsensusExcludeLossPTB(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha = config.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        self.mask = torch.ones((config.models_num, config.models_num), requires_grad=False)
        for k in range(self.models_num):
            self.mask[k, k] = 0
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def __call__(self, index, inputs, hiddenes, targets):
        logits_list = [0 for _ in range(self.models_num)]
        q_softmaxes = [0 for _ in range(self.models_num)]
        ce_loss=0
        kl_loss = 0
        for k in range(self.models_num):
            logits_list[k], hiddenes[k] = self.models[k](inputs, hiddenes[k])
            ce_loss += F.cross_entropy(logits_list[k], targets)
        for k in range(self.models_num):
            q_masked = self.q*self.mask[k] - (1 - self.mask[k])*1e10
            q_softmaxes[k] = F.softmax(q_masked, dim=0) if self.learnable_q else F.softmax(torch.zeros(self.models_num), dim=0)
            for l in range(self.models_num):
                if self.detach:
                    kl_loss += q_softmaxes[k][l] * (kl_div_logits(logits_list[k], logits_list[l].detach(), self.T) + 
                                               kl_div_logits(logits_list[l].detach(), logits_list[k], self.T))
                else:
                    kl_loss += q_softmaxes[k][l] * (kl_div_logits(logits_list[k], logits_list[l], self.T) + 
                                               kl_div_logits(logits_list[l], logits_list[k], self.T))
        loss = self.alpha*kl_loss + ce_loss
        return loss, logits_list[index], logits_list, hiddenes



class ClassifierConsensusForthLossPTB(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha = config.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        self.mask = torch.ones((config.models_num, config.models_num), requires_grad=False)
        for i in range(self.models_num):
            self.mask[i, i] = 0
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, hiddenes, targets):
        ce_loss=0
        kl_loss=0
        logits_list = [0 for _ in range(self.models_num)]
        for k in range(self.models_num):
            logits_list[k], hiddenes[k] = self.models[k](inputs, hiddenes[k])
        models_pred = torch.stack([F.log_softmax(logits_list[k], dim=-1) for k in range(self.models_num)])
        q_logits =  torch.stack([F.log_softmax(self.q*self.mask[k] - (1 - self.mask[k])*1e10, dim=0).cuda().view(self.models_num, 1, 1) for k in range(self.models_num)])
        models_pred_multi = [0 for _ in range(self.models_num)]
        for k in range(self.models_num):
            models_pred_multi[k] = models_pred.detach() + q_logits[k]
        ensemble_logits = torch.stack([torch.logsumexp(models_pred_multi[k], dim=0) for k in range(self.models_num)])
        ensemble_logits_normalized = F.log_softmax(ensemble_logits, dim=-1)
        for k in range(self.models_num):
            kl_loss+=kl_div_logits(logits_list[k], ensemble_logits_normalized[k], self.T)
            ce_loss+=F.cross_entropy(logits_list[k], targets)
        # loss = self.alpha*kl_loss + ce_loss
        loss = ce_loss
        return loss, models_pred[index], models_pred, hiddenes


class ClassifierConsensusFifthLossPTB(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha = config.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        self.mask = torch.ones((config.models_num, config.models_num), requires_grad=False)
        for i in range(self.models_num):
            self.mask[i, i] = 0
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, hiddenes, targets):
        logits_list = [0 for _ in range(self.models_num)]
        for k in range(self.models_num):
            logits_list[k], hiddenes[k] = self.models[k](inputs, hiddenes[k])
        teacher_loss=F.cross_entropy(logits_list[0], targets)+ \
                    self.alpha*(kl_div_logits(logits_list[0], logits_list[1].detach(), self.T) + \
                    kl_div_logits(logits_list[1].detach(), logits_list[0], self.T))
        student_loss=kl_div_logits(logits_list[1], logits_list[0].detach(), self.T) + \
                    kl_div_logits(logits_list[0].detach(), logits_list[1], self.T)
        loss = teacher_loss + student_loss
        return loss, logits_list[index], logits_list, hiddenes


class ClassifierConsensusTogetherLoss(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        logits_list = [0 for _ in range(self.models_num)]
        ce_loss=0
        for i in range(self.models_num):
            logits_list[i] = self.models[i](inputs)
            ce_loss += F.cross_entropy(logits_list[i], targets)
        kl_loss = 0
        q_softmax = F.softmax(self.q, dim=0) if self.learnable_q else F.softmax(torch.zeros(self.models_num), dim=0)
        if self.detach:
            weighted_prob = sum(F.softmax(logits.detach(), dim=0) * weight for logits, weight in zip(logits_list, q_softmax)) 
        else:
            weighted_prob = sum(F.softmax(logits, dim=0) * weight for logits, weight in zip(logits_list, q_softmax)) 
        normalized_weighted_prob = weighted_prob / weighted_prob.sum(dim=1, keepdim=True)
        for i in range(self.models_num):
            kl_loss += (kl_div_softmax(logits_list[i], torch.log(normalized_weighted_prob), self.T) + 
                        kl_div_softmax( torch.log(normalized_weighted_prob), logits_list[i], self.T))
        loss = self.alpha*kl_loss + ce_loss
        return loss, logits_list[index], logits_list



class ClassifierConsensusTogetherDirectLoss(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        ce_loss=0
        kl_loss=0
        q_logits = (F.log_softmax(self.q, dim=0) if self.learnable_q else \
                             F.log_softmax(torch.zeros(self.models_num), dim=0)).cuda().view(self.models_num, 1, 1)
        models_pred = torch.stack([F.log_softmax(self.models[i](inputs), dim=-1) for i in range(self.models_num)])
        if self.detach:
            models_pred_add = models_pred.detach() + q_logits
            models_pred_sum = torch.logsumexp(models_pred.detach(), dim = -1, keepdim=True)
        else:
            models_pred_add = models_pred + q_logits
            models_pred_sum = torch.logsumexp(models_pred, dim = -1, keepdim=True)
        models_pred_weighted = models_pred_add - models_pred_sum
        ensemble_weighted_logits = torch.logsumexp(models_pred_weighted, dim=0)
        ensemble_weighted_logits_normalized = F.log_softmax(ensemble_weighted_logits, dim=-1)
        for i in range(self.models_num):
            kl_loss+=kl_div_logits(models_pred[i], ensemble_weighted_logits_normalized, self.T)
            ce_loss+=F.cross_entropy(models_pred[i], targets)
        loss = self.alpha*kl_loss + ce_loss

        return loss, models_pred[index], models_pred




class ClassifierConsensusForthLoss(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        self.mask = torch.ones((config.models_num, config.models_num), requires_grad=False)
        for i in range(self.models_num):
            self.mask[i, i] = 0
        self.T = 1
        self.detach = config.detach
        self.learnable_q = config.learnable_q
        print(f'detach: {self.detach}, learnable_q: {self.learnable_q}')

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        ce_loss=0
        kl_loss=0
        q_logits =  torch.stack([F.log_softmax(self.q*self.mask[i] - (1 - self.mask[i])*1e10, dim=0).cuda().view(self.models_num, 1, 1) for i in range(self.models_num)])
        models_pred = torch.stack([F.log_softmax(self.models[i](inputs), dim=-1) for i in range(self.models_num)])
        models_pred_multi = [0 for _ in range(self.models_num)]
        for k in range(self.models_num):
            models_pred_multi[k] = models_pred.detach() + q_logits[k]
        ensemble_logits = torch.stack([torch.logsumexp(models_pred_multi[k], dim=0) for k in range(self.models_num)])
        ensemble_logits_normalized = F.log_softmax(ensemble_logits, dim=-1)
        for k in range(self.models_num):
            kl_loss+=kl_div_logits(models_pred[k], ensemble_logits_normalized[k], self.T)
            ce_loss+=F.cross_entropy(models_pred[k], targets)
        loss = self.alpha*kl_loss + ce_loss
        return loss, models_pred[index], models_pred



class ClassifierConsensusAllLoss_detach(object):
    def __init__(self, models, config):
        self.models= models
        self.models_num = config.models_num
        self.alpha_steps = config.alpha_steps
        self.alpha_values = config.alpha_values
        self.alpha = 2 / self.models_num if config.loss.alpha == -1 else config.loss.alpha
        self.q = torch.nn.Parameter(torch.zeros(config.models_num))
        # self.alpha = torch.nn.Parameter(torch.tensor(alpha))

    def calculate_alpha(self, epoch):
        self.steps

    def __call__(self, index, inputs, targets):
        logits_list = [0 for _ in range(self.models_num)]
        models_dists = [0 for _ in range(self.models_num)]
        net_logits = self.models[index](inputs)
        net_dists = Categorical(logits = net_logits)
        ce_loss = F.cross_entropy(net_logits, targets)
        for i in range(self.models_num):
            if i == index:
                logits_list[i] = net_logits
            else:
                logits_list[i] = self.models[i](inputs).detach()
            models_dists[i] = Categorical(logits = logits_list[i])
        kl_loss = 0
        q_softmax = F.softmax(self.q, dim=0)
        for i in range(self.models_num):
            # if i == index:
            #     continue
            kl_loss += q_softmax[i] * (kl_divergence(models_dists[i], net_dists).mean() + kl_divergence(net_dists, models_dists[i]).mean())
        # alpha = torch.sigmoid(self.alpha)
        loss = self.alpha*kl_loss + ce_loss
        return loss, net_logits, logits_list


class ClassifierStudentLoss(object):
    def __init__(self, student_model, base_loss, alpha=0.9):
        self.student = student_model
        self.base_loss = base_loss
        self.alpha = alpha

    def __call__(self, inputs, targets, teacher_logits, temp=None):
        real_batch_size = targets.size(0)
        student_logits = self.student(inputs)
        hard_loss = F.cross_entropy(student_logits[:real_batch_size], targets)
        # temp = torch.ones_like(student_logits) if temp is None else temp.unsqueeze(-1)
        temp = torch.ones_like(student_logits) if temp is None else temp
        soft_loss = self.base_loss(teacher_logits, student_logits, temp)
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return loss, student_logits


def reduce_ensemble_logits(teacher_logits):
    assert teacher_logits.dim() == 3
    teacher_logits = F.log_softmax(teacher_logits, dim=-1)
    n_teachers = len(teacher_logits)
    return torch.logsumexp(teacher_logits, dim=1) - math.log(n_teachers)


def compete_logits(teacher_logits, targets):
    teacher_seperate = torch.unbind(teacher_logits, dim=1)
    t_losses=[]
    for t in teacher_seperate:
        t_loss=F.cross_entropy(t, targets)
        t_losses.append(t_loss)
    min_index=torch.argmin(torch.Tensor(t_losses))
    return teacher_seperate[min_index]


class BaseClassificationDistillationLoss(ABC):
    """Abstract class that defines interface for distillation losses.
    """

    def __call__(self, teacher_logits, student_logits, temp=1.):
        """Evaluate loss.

        :param teacher_logits: tensor of teacher model logits of size
            [num_teachers, batch_size, num_classes] or [batch_size, num_classes]
        :param student_logits: tensor of student model logits of size
            [batch_size, num_classes]
        :param temp: temperature to apply to the teacher logits
        :return: scalar loss value
        """
        teacher_logits = self._reduce_teacher_predictions(teacher_logits)
        assert teacher_logits.shape == student_logits.shape, \
            "Shape mismatch: teacher logits" \
            "have shape {} and student logits have shape {}".format(
                    teacher_logits.shape, student_logits.shape)
        return self.teacher_student_loss(teacher_logits, student_logits, temp)

    @staticmethod
    def _reduce_teacher_predictions(teacher_logits):
        if len(teacher_logits.shape) == 3:
            return reduce_ensemble_logits(teacher_logits)
        return teacher_logits

    @staticmethod
    @abstractmethod
    def teacher_student_loss(teacher_logits, student_logits, temp):
        raise NotImplementedError


class TeacherStudentKLLoss(BaseClassificationDistillationLoss):
    """KL loss between the teacher and student predictions.
    """
    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits, temp):
        teacher_dist = Categorical(logits=teacher_logits / temp)
        student_dist = Categorical(logits=student_logits / temp)

        return kl_divergence(teacher_dist, student_dist).mean()


class SymmetrizedKLLoss(BaseClassificationDistillationLoss):
    """Symmetrized KL loss.
    """
    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits, temp):
        teacher_dist = Categorical(logits=teacher_logits / temp)
        student_dist = Categorical(logits=student_logits / temp)

        kl_p_q = kl_divergence(teacher_dist, student_dist)
        kl_q_p = kl_divergence(student_dist, teacher_dist)
        return kl_p_q.mean() + kl_q_p.mean()


class BrierLoss(BaseClassificationDistillationLoss):
    """Brier loss.

    Note: error is averaged both over the classes and data.
    """
    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits, temp):
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        student_probs = F.softmax(student_logits / temp, dim=-1)
        return F.mse_loss(student_probs, teacher_probs)


class AveragedSymmetrizedKLLoss(BaseClassificationDistillationLoss):
    """Symmetrized KL averaged over teacher models.

    Here, instead of using the ensemble, we compute the average KL to each of
    the teacher models. This is the loss that we had implemented originally.
    """

    def __call__(cls, teacher_logits, student_logits, temp=1.):
        assert teacher_logits.size(0) == student_logits.size(0)
        assert teacher_logits.size(-1) == student_logits.size(-1)
        return cls.teacher_student_loss(teacher_logits.transpose(1, 0), student_logits, temp)

    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits, temp):
        teacher_dist = Categorical(logits=teacher_logits / temp)
        student_dist = Categorical(logits=student_logits / temp)

        kl = kl_divergence(teacher_dist, student_dist).mean()
        reversed_kl = kl_divergence(student_dist, teacher_dist).mean()

        return kl + reversed_kl


class TeacherStudentHardCrossEntLoss(object):
    """
    Standard cross-entropy loss w.r.t. the hard teacher labels
    """
    def __init__(self, corruption_ratio=0., mode='argmax', **kwargs):
        super().__init__()
        self.corruption_ratio = corruption_ratio
        self.mode = mode

    def __call__(self, teacher_logits, student_logits, temp):
        if teacher_logits.dim() == 3:
            teacher_logits = reduce_ensemble_logits(teacher_logits)

        batch_size, num_classes = teacher_logits.shape
        if self.mode == 'argmax':
            teacher_labels = torch.argmax(teacher_logits, dim=-1)
        elif self.mode == 'sample':
            teacher_dist = Categorical(logits=teacher_logits / temp)
            teacher_labels = teacher_dist.sample()

        num_corrupted = int(batch_size * self.corruption_ratio)
        if num_corrupted > 0:
            rand_labels = torch.randint(0, num_classes, (num_corrupted,), device=teacher_labels.device)
            corrupt_idxs = torch.randint(0, batch_size, (num_corrupted,))
            teacher_labels[corrupt_idxs] = rand_labels

        student_logp = F.log_softmax(student_logits / temp, dim=-1)
        student_logp = student_logp[torch.arange(batch_size), teacher_labels]
        temp = temp[torch.arange(batch_size), teacher_labels]
        loss = -(temp ** 2 * student_logp).mean()
        return loss


class TeacherStudentFwdCrossEntLoss(object):
    """Soft teacher/student cross entropy loss from [Hinton et al (2015)]
        (https://arxiv.org/abs/1503.02531)
    """
    def __call__(self, teacher_logits, student_logits, temp):
        if teacher_logits.dim() == 3:
            teacher_logits = reduce_ensemble_logits(teacher_logits)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        student_logp = F.log_softmax(student_logits / temp, dim=-1)
        loss = -(temp ** 2 * teacher_probs * student_logp).sum(-1).mean()
        return loss


class TeacherStudentCvxCrossEnt(object):
    def __init__(self, T_max, beta=0.5):
        self.hard_loss_fn = TeacherStudentHardCrossEntLoss(corruption_ratio=0.)
        self.soft_loss_fn = TeacherStudentFwdCrossEntLoss()
        self._init_beta = beta
        self.beta = beta
        self.t_max = T_max

    def __call__(self, teacher_logits, student_logits, temp):
        hard_loss = self.hard_loss_fn(teacher_logits, student_logits, temp)
        soft_loss = self.soft_loss_fn(teacher_logits, student_logits, temp)
        cvx_loss = self.beta * hard_loss + (1 - self.beta) * soft_loss
        return cvx_loss

    def step(self):
        next_beta = self.beta - self._init_beta / self.t_max
        next_beta = max(next_beta, 0.)
        self.beta = next_beta


class TeacherStudentRevCrossEntLoss(object):
    """Soft teacher/student cross entropy loss from [Hinton et al (2015)]
        (https://arxiv.org/abs/1503.02531)
    """
    def __call__(self, teacher_logits, student_logits, temp):
        if teacher_logits.dim() == 3:
            teacher_logits = reduce_ensemble_logits(teacher_logits)
        teacher_logp = F.log_softmax(teacher_logits / temp, dim=-1)
        student_probs = F.softmax(student_logits / temp, dim=-1)
        loss = -(temp ** 2 * student_probs * teacher_logp).sum(-1).mean()
        return loss
