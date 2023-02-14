''' MuFAN in PyTorch.

Mainly adapted from https://github.com/phquang/DualNet/blob/main/model/dualnet.py

Reference:
[1] Jung, Dahuin, et al. "New Insights for the Stability-Plasticity Dilemma in Online Continual Learning."
    International Conference on Learning Representations 2023.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import random

from .encoder import Encoder_Proj
from .resnet_mufan import ResNet18_MuFAN

class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        # setup network
        self.is_cifar = any(x in str(args.data_file) for x in ['svhn', 'cifar', 'mini', 'core'])

        self.net = ResNet18_MuFAN(num_classes=n_outputs)
        if args.cuda:
            self.net = self.net.cuda()

        if 'mini' in args.data_file:
            self.feature_network = Encoder_Proj(coco_ssdlite=True)
        else:
            self.feature_network = Encoder_Proj()

        if args.cuda:
            self.feature_network = self.feature_network.cuda()
        self.feature_network = self.feature_network.train(False)

        self.transform_train = nn.Sequential(
            transforms.RandomCrop(128, padding=16),
            transforms.RandomHorizontalFlip())

        self.lr = args.lr
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)

        # setup losses
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss()

        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs

        # setup memories
        self.current_task = 0
        self.n_memories = args.n_memories

        if 'svhn' in args.data_file or 'cifar' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 32, 32)
        elif 'mini' in args.data_file or 'core' in args.data_file:
            self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 128, 128)

        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task)
        self.mem = {}

        if args.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.mem_feat = self.mem_feat.cuda()

        self.mem_cnt = 0
        self.n_memories = args.n_memories
        self.n_outputs = n_outputs

        self.bsz = args.batch_size
        self.sz = args.replay_batch_size
        self.inner_steps = args.inner_steps
        self.cuda = args.cuda

        self.lambda_ce = args.lambda_ce
        self.lambda_ctn = args.lambda_ctn
        self.lambda_csd = args.lambda_csd

        self.temp = args.ctn_temp
        self.temp_student = args.csd_student_temp
        self.temp_teacher = args.csd_teacher_temp

        self.num_distill = args.num_distill

    def on_epoch_end(self):
        pass

    def soft_cross_entropy(self, student_logit, teacher_logit):
        return -(teacher_logit * torch.nn.functional.log_softmax(student_logit, 1)).sum() / student_logit.shape[0]

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t, return_feat=False):
        features = self.feature_network(x)
        output = self.net(features['0'])

        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t)

            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def memory_sampling(self, t):
        mem_x = self.memx[:t, :]
        mem_y = self.memy[:t, :]
        mem_feat = self.mem_feat[:t, :]
        sz = min(self.n_memories, self.sz)
        idx = np.random.choice(t * self.n_memories, sz, False)
        t_idx = torch.from_numpy(idx // self.n_memories)
        s_idx = torch.from_numpy(idx % self.n_memories)

        if self.cuda:
            offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
        xx = mem_x[t_idx, s_idx]
        yy = mem_y[t_idx, s_idx] - offsets[:, 0]
        feat = mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task)
        for j in range(mask.size(0)):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        return xx, yy, feat, mask.long().cuda() if self.cuda else mask.long()

    def observe(self, x, t, y):
        self.net.train()
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memx[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        self.memy[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)

            xx = self.transform_train(self.memx[tt])
            if self.cuda:
                xx = xx.cuda()
            out = self.forward(xx, tt, True)
            self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1).data.clone()

            # Section 3.2 Cross-task structure-wise distillation loss of MuFAN in PyTorch (teacher part).
            if t > 1:
                self.num_list = random.sample(range(0, self.n_memories), self.num_distill)

                total_t_emb = []
                for i in range(0, tt + 1):
                    offset1, offset2 = self.compute_offsets(i)
                    xx = self.transform_train(self.memx[i, self.num_list])
                    if self.cuda:
                        xx = xx.cuda()
                    t_emb = self.forward(xx, i, True)
                    t_emb = nn.functional.normalize(t_emb[:, offset1:offset2], dim=1)
                    total_t_emb.append(t_emb)
                total_t_emb = torch.cat(total_t_emb, dim=0)

                self.logit_tea = []
                for i in range(1, tt+1):
                    logit_tea = torch.einsum('nc,ck->nk', [total_t_emb[(i-1) * self.num_distill:i * self.num_distill].detach(),
                                                           total_t_emb[i * self.num_distill:(i + 1) * self.num_distill].T.detach()])
                    self.logit_tea.append(nn.functional.softmax(logit_tea / self.temp_teacher, dim=1))

        self.current_task = t
        for _ in range(self.inner_steps):
            self.net.zero_grad()
            # loss1 = torch.tensor(0.).cuda()
            loss2 = torch.tensor(0.)
            loss3 = torch.tensor(0.)
            loss4 = torch.tensor(0.)

            if self.cuda:
                loss2 = loss2.cuda()
                loss3 = loss3.cuda()
                loss4 = loss4.cuda()

            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x, t, True)
            loss1 = self.lambda_ce * self.ce(pred[:, offset1:offset2], y - offset1)
            if t > 0:
                xx, yy, target, mask = self.memory_sampling(t)
                xx = self.transform_train(xx)
                if self.cuda:
                    xx = xx.cuda()
                xx_features = self.feature_network(xx)
                pred_ = self.net(xx_features['0'])
                pred = torch.gather(pred_, 1, mask)
                loss2 += self.ce(pred, yy)
                loss3 = self.lambda_ctn * self.kl(F.log_softmax(pred / self.temp, dim=1), target)
            if t > 1:
                # Section 3.2 Cross-task structure-wise distillation loss of MuFAN in PyTorch (student part).
                total_t_student_emb = []
                for i in range(0, t):
                    offset1, offset2 = self.compute_offsets(i)
                    xx = self.transform_train(self.memx[i, self.num_list])
                    if self.cuda:
                        xx = xx.cuda()
                    t_emb = self.forward(xx, i, True)
                    t_emb = nn.functional.normalize(t_emb[:, offset1:offset2], dim=1)
                    total_t_student_emb.append(t_emb)
                total_t_student_emb = torch.cat(total_t_student_emb, dim=0)

                self.logit_stu = []
                for i in range(1, t):
                    logit_stu = torch.einsum('nc,ck->nk', [total_t_student_emb[(i-1) * self.num_distill:i * self.num_distill],
                                                           total_t_student_emb[i * self.num_distill:(i + 1) * self.num_distill].T])
                    self.logit_stu.append(logit_stu / self.temp_student)

                    loss4 += self.lambda_csd * self.soft_cross_entropy(self.logit_stu[i-1], self.logit_tea[i-1])

            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            self.opt.step()

        return loss.item()
