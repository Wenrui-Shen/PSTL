from config import *
from model import *
from dataset import DataSet, Feeder_semi
from logger import Log

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import random
from tqdm import tqdm
from einops import rearrange, repeat
from math import pi, cos
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from module.gcn.st_gcn import Model
from module.gatr_encoder import SkeletonGATrEncoder


def strip_torchrun_args():
    cleaned = [sys.argv[0]]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ('--local-rank', '--local_rank'):
            skip_next = True
            continue
        if arg.startswith('--local-rank=') or arg.startswith('--local_rank='):
            continue
        cleaned.append(arg)
    sys.argv = cleaned


strip_torchrun_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
setup_seed(1)


def setup_distributed():
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', '0'))


def is_main_process():
    return get_rank() == 0


def progress(loader):
    return tqdm(loader, disable=not is_main_process())


def wrap_ddp(module):
    if not is_distributed():
        return module
    return DDP(module, device_ids=[get_local_rank()], output_device=get_local_rank())


def unwrap_model(module):
    return module.module if isinstance(module, DDP) else module


def normalize_gatr_state_dict_keys(state_dict):
    replacements = {
        '.attention.qkv_q_linear.': '.attention.qkv_module.q_linear.',
        '.attention.qkv_k_linear.': '.attention.qkv_module.k_linear.',
        '.attention.qkv_v_linear.': '.attention.qkv_module.v_linear.',
    }
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        normalized[new_key] = value
    return normalized


class NoOpLog:
    def update_batch(self, name, value):
        pass

    def update_epoch(self, *args, **kwargs):
        pass


@ex.capture
def get_pretrain_accumulation_steps(
    auto_accumulate_pretrain,
    base_global_batch_size,
    batch_size,
):
    if not auto_accumulate_pretrain:
        return 1
    per_step_global_batch_size = batch_size * get_world_size()
    accumulation_steps = max(
        1,
        (base_global_batch_size + per_step_global_batch_size - 1)
        // per_step_global_batch_size)
    if is_main_process():
        effective_global_batch_size = per_step_global_batch_size * accumulation_steps
        print(
            'Pretrain accumulation steps: {} '
            '(per_step_global_batch_size={}, effective_global_batch_size={}, target_global_batch_size={})'
            .format(
                accumulation_steps,
                per_step_global_batch_size,
                effective_global_batch_size,
                base_global_batch_size)
        )
    return accumulation_steps


@ex.capture
def get_pretrain_lr(
    pretrain_lr,
    auto_scale_pretrain_lr,
    base_pretrain_lr,
    base_global_batch_size,
    batch_size,
    pretrain_accumulation_steps=None,
):
    if not auto_scale_pretrain_lr:
        return pretrain_lr
    if pretrain_accumulation_steps is None:
        pretrain_accumulation_steps = get_pretrain_accumulation_steps()
    global_batch_size = batch_size * get_world_size() * pretrain_accumulation_steps
    scaled_lr = base_pretrain_lr * global_batch_size / base_global_batch_size
    if is_main_process():
        print(
            'Auto-scaled pretrain lr: {} '
            '(effective_global_batch_size={}, base_lr={}, base_global_batch_size={})'
            .format(scaled_lr, global_batch_size, base_pretrain_lr, base_global_batch_size)
        )
    return scaled_lr


@ex.capture
def build_encoder(
    encoder_type,
    in_channels,
    hidden_channels,
    hidden_dim,
    hidden_size,
    dropout,
    graph_args,
    edge_importance_weighting,
    max_frame,
    joint_num,
    person_num,
    gatr_num_blocks,
    gatr_hidden_mv_channels,
    gatr_point_s_channels,
    gatr_hidden_s_channels,
    gatr_out_mv_channels,
    gatr_num_heads,
    gatr_checkpoint_blocks,
):
    if encoder_type == 'stgcn':
        return Model(in_channels=in_channels, hidden_channels=hidden_channels,
                     hidden_dim=hidden_dim, dropout=dropout,
                     graph_args=graph_args,
                     edge_importance_weighting=edge_importance_weighting)
    if encoder_type == 'gatr':
        return SkeletonGATrEncoder(hidden_size=hidden_size,
                                   num_frame=max_frame,
                                   num_joint=joint_num,
                                   num_person=person_num,
                                   out_mv_channels=gatr_out_mv_channels,
                                   hidden_mv_channels=gatr_hidden_mv_channels,
                                   in_s_channels=gatr_point_s_channels,
                                   hidden_s_channels=gatr_hidden_s_channels,
                                   num_blocks=gatr_num_blocks,
                                   num_heads=gatr_num_heads,
                                   checkpoint_blocks=gatr_checkpoint_blocks)
    raise ValueError('Invalid encoder_type: {}'.format(encoder_type))

class BaseProcessor:

    @ex.capture
    def configure_backend(self, use_cudnn):
        torch.backends.cudnn.enabled = use_cudnn

    @ex.capture
    def load_data(self,train_list,train_label,test_list,test_label,batch_size,label_percent,num_workers):
        self.dataset = dict()
        self.data_loader = dict()

        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)
        # self.dataset['semi'] = Feeder_semi(train_list, train_label, label_percent)
        self.train_sampler = DistributedSampler(self.dataset['train'], shuffle=True) if is_distributed() else None

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False)
        
        # self.data_loader['semi'] = torch.utils.data.DataLoader(
        #     dataset=self.dataset['semi'],
        #     batch_size=batch_size,
        #     num_workers=32,
        #     shuffle=True)
        
    def load_weights(self, model=None, weight_path=None):
        if weight_path:
            pretrained_dict = torch.load(weight_path, map_location='cuda:{}'.format(get_local_rank()))
            if isinstance(pretrained_dict, dict) and 'encoder' in pretrained_dict:
                pretrained_dict = pretrained_dict['encoder']
            pretrained_dict = {
                key.replace('module.', '', 1): value
                for key, value in pretrained_dict.items()
            }
            pretrained_dict = normalize_gatr_state_dict_keys(pretrained_dict)
            model.load_state_dict(pretrained_dict)

    def initialize(self):
        self.start_epoch = 0
        self.configure_backend()
        self.load_data()
        self.load_model()
        self.load_optim()
        self.load_resume()
        self.log = Log() if is_main_process() else NoOpLog()

    def set_sampler_epoch(self, epoch):
        if hasattr(self, 'train_sampler') and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def load_resume(self):
        pass
    
    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.set_sampler_epoch(epoch)
            self.train_epoch()
            self.test_epoch()
    
    def adjust_learning_rate(self, optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch=10):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_model(self):
        
        pass

    def start(self):
        self.initialize()
        self.optimize()
        self.save_model()

# %%
class RecognitionProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,weight_path,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting):
        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        self.classifier = Linear().cuda()
        self.load_weights(self.encoder, weight_path)
        self.classifier = wrap_ddp(self.classifier)
    
    @ex.capture
    def load_optim(self, lp_lr, lp_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
             lr=lp_lr,
             )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, lp_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self, epoch, lp_epoch, lp_lr):
        self.encoder.eval()
        self.classifier.train()

        loader = self.data_loader['train']
        for data, label in progress(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data)
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label):

        Z = self.encoder(data)
        Z = Z.detach()
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/train/cls_acc", acc.item())
        self.log.update_batch("log/train/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, result_path, label_path, save_lp):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = result_path + str(epoch) + '_result.pkl'

        loader = self.data_loader['test']
        for data, label in progress(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data)

            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_lp and is_main_process():
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)

    def save_model(self):
        
        pass
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            if is_main_process():
                print("epoch:",epoch)
            self.epoch = epoch
            self.set_sampler_epoch(epoch)
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class SemiProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,weight_path,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting):
        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        self.classifier = Linear().cuda()
        self.load_weights(self.encoder, weight_path)
        self.encoder = wrap_ddp(self.encoder)
        self.classifier = wrap_ddp(self.classifier)
    
    @ex.capture
    def load_optim(self, ft_lr, ft_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
            lr=ft_lr,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, ft_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['semi']
        for data, label in progress(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data)
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label): 

        Z = self.encoder(data)
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/semi_train/cls_acc", acc.item())
        self.log.update_batch("log/semi_train/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, result_path, label_path, save_semi=True):
        self.encoder.eval()
        self.classifier.eval()

        result_list = []
        label_list = []
        r_path = result_path + str(epoch) + '_semi10_result.pkl'

        loader = self.data_loader['test']
        for data, label in progress(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data)
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/semi_test/cls_acc", acc.item())
            self.log.update_batch("log/semi_test/cls_loss", loss.item())

        if save_semi and is_main_process():
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            if is_main_process():
                print("epoch:",epoch)
            self.set_sampler_epoch(epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class FTProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,weight_path,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting):
        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        self.classifier = Linear().cuda()
        self.load_weights(self.encoder, weight_path)
        self.encoder = wrap_ddp(self.encoder)
        self.classifier = wrap_ddp(self.classifier)
    
    @ex.capture
    def load_optim(self, ft_lr, ft_epoch):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters()}],
            lr=ft_lr,
            )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, ft_epoch)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['train']
        for data, label in progress(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = get_stream(data)
            loss = self.train_batch(data, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label):

        Z = self.encoder(data)
        predict = self.classifier(Z)
        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()
        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/finetune/cls_acc", acc.item())
        self.log.update_batch("log/finetune/cls_loss", loss.item())

        return loss

    @ex.capture
    def test_epoch(self, epoch, result_path, label_path, save_finetune):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = result_path + str(epoch) + '_finetune_result.pkl'

        loader = self.data_loader['test']
        for data, label in progress(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            label_list.append(label)
            data = get_stream(data)
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_finetune and is_main_process():
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            if is_main_process():
                print("epoch:",epoch)
            self.set_sampler_epoch(epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class BTProcessor(BaseProcessor):
    
    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,dropout,
                    graph_args,edge_importance_weighting):

        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        self.btwins_head = BTwins().cuda()
        self.encoder = wrap_ddp(self.encoder)
        self.btwins_head = wrap_ddp(self.btwins_head)

    @ex.capture
    def load_optim(self, pretrain_epoch, weight_decay):
        self.pretrain_accumulation_steps = get_pretrain_accumulation_steps()
        self.pretrain_lr = get_pretrain_lr(
            pretrain_accumulation_steps=self.pretrain_accumulation_steps)
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.btwins_head.parameters()},
            ], 
            weight_decay=weight_decay,
            lr=self.pretrain_lr)

    def btwins_batch(self, feat1, feat2, mode):
        BTloss = self.btwins_head(feat1, feat2)
        BTloss = torch.mean(BTloss)
        self.log.update_batch("log/pretrain/"+mode+"_bt_loss", BTloss.item())
        return BTloss

    @ex.capture
    def infonce_batch(self, anchor_feat, positive_feat, negative_feat, mode, infonce_temperature):
        loss = self.btwins_head(
            anchor_feat,
            positive_feat,
            negative_feat,
            loss_type='infonce',
            temperature=infonce_temperature)
        self.log.update_batch("log/pretrain/"+mode+"_infonce_loss", loss.item())
        return loss

    @ex.capture
    def symmetric_infonce_batch(self, anchor_feat, positive_feat, negative_feat, mode, infonce_temperature):
        loss = self.btwins_head(
            anchor_feat,
            positive_feat,
            negative_feat,
            loss_type='symmetric_infonce',
            temperature=infonce_temperature)
        self.log.update_batch("log/pretrain/"+mode+"_infonce_loss", loss.item())
        return loss

    @ex.capture
    def load_resume(self, resume, resume_path):
        if not resume:
            return
        if not os.path.exists(resume_path):
            raise FileNotFoundError('Resume checkpoint not found: {}'.format(resume_path))
        checkpoint = torch.load(
            resume_path,
            map_location='cuda:{}'.format(get_local_rank()))
        unwrap_model(self.encoder).load_state_dict(checkpoint['encoder'])
        unwrap_model(self.btwins_head).load_state_dict(checkpoint['btwins_head'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        if is_main_process():
            print('Resumed from {} at epoch {}'.format(resume_path, self.start_epoch))

    @ex.capture
    def save_checkpoint(self, epoch, checkpoint_path):
        if not is_main_process():
            return
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'encoder': unwrap_model(self.encoder).state_dict(),
            'btwins_head': unwrap_model(self.btwins_head).state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)

    def skeleton_aug(self, data):
        data = shear(crop(data))
        data = random_rotate(data)
        data = random_spatial_flip(data)
        return data

    def e3_aug(self, data):
        return random_e3_transform(data)

    def non_e3_aug(self, data):
        data = shear(crop(data))
        data = random_spatial_flip(data)
        return data

    @ex.capture
    def train_epoch(self, epoch, pretrain_epoch, pretrain_lr):
        self.encoder.train()
        self.btwins_head.train()

        loader = self.data_loader['train']
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=pretrain_epoch, lr_max=self.pretrain_lr)

        accumulation_steps = getattr(self, 'pretrain_accumulation_steps', 1)
        self.optimizer.zero_grad()

        for batch_idx, (data, label) in enumerate(progress(loader)):
            # load data
            n,c,t,v,m = data.shape
            data = data.type(torch.FloatTensor).cuda()
            data = get_stream(data)

            # get ignore joint
            ignore_joint = central_spacial_mask()

            # input1
            input1 = self.skeleton_aug(data)
            feat1 = self.encoder(input1)

            # input2
            input2 = self.skeleton_aug(data)
            # MATM
            input2 = motion_att_temp_mask(input2)
            feat2 = self.encoder(input2)

            # input3
            input3 = self.skeleton_aug(data)
            # CSM
            feat3 = self.encoder(input3, ignore_joint)

            # loss
            loss_bt1 = self.btwins_batch(feat1, feat2, mode='temp_mask')
            loss_bt2 = self.btwins_batch(feat1, feat3, mode='joint_mask')

            loss = (loss_bt1 + loss_bt2) / accumulation_steps

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                self.optimizer.step()
                self.optimizer.zero_grad()
    @ex.capture
    def save_model(self, epoch,version):
        if is_main_process():
            os.makedirs('output/weight', exist_ok=True)
            torch.save(unwrap_model(self.encoder).state_dict(), f"output/weight/v"+version+"_epoch_"+str(epoch+1)+"_pretrain.pt")
        
    @ex.capture
    def optimize(self, pretrain_epoch, checkpoint_interval):
        for epoch in range(self.start_epoch, pretrain_epoch):
            if is_main_process():
                print("epoch:",epoch)
            self.epoch = epoch
            self.set_sampler_epoch(epoch)
            self.train_epoch(epoch=epoch)
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            if epoch+1 == pretrain_epoch:
                self.save_model(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,pretrain_epoch,lr=lr)
            
    @ex.capture
    def start(self):
        self.initialize()
        self.optimize()


class GATrProcessor(BTProcessor):

    def load_model(self):
        self.encoder = build_encoder(encoder_type='gatr')
        self.encoder = self.encoder.cuda()
        self.btwins_head = BTwins().cuda()
        self.encoder = wrap_ddp(self.encoder)
        self.btwins_head = wrap_ddp(self.btwins_head)

    @ex.capture
    def train_epoch(self, epoch, pretrain_epoch, pretrain_lr):
        self.encoder.train()
        self.btwins_head.train()

        loader = self.data_loader['train']
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=pretrain_epoch, lr_max=self.pretrain_lr)

        accumulation_steps = getattr(self, 'pretrain_accumulation_steps', 1)
        self.optimizer.zero_grad()

        for batch_idx, (data, label) in enumerate(progress(loader)):
            # load data
            n,c,t,v,m = data.shape
            data = data.type(torch.FloatTensor).cuda()
            data = get_stream(data)

            # Raw view is the anchor; E(3) view is positive; non-E(3) view is negative.
            input_raw = data
            input_e3 = self.e3_aug(data)
            input_non_e3 = self.non_e3_aug(data)

            feat_raw = self.encoder(input_raw)
            feat_e3 = self.encoder(input_e3)
            with torch.no_grad():
                feat_non_e3 = self.encoder(input_non_e3)

            loss = self.symmetric_infonce_batch(
                feat_raw,
                feat_e3,
                feat_non_e3,
                mode='raw_e3_non_e3') / accumulation_steps

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                self.optimizer.step()
                self.optimizer.zero_grad()


# %%
@ex.automain
def main(train_mode):
    setup_distributed()
    try:
        if "gatr" in train_mode:
            p = GATrProcessor()
        elif "pretrain" in train_mode:
            p = BTProcessor()
        elif "lp" in train_mode:
            p = RecognitionProcessor()
        elif "finetune" in train_mode:
            p = FTProcessor()
        elif "semi" in train_mode:
            p = SemiProcessor()
        else:
            raise ValueError('train_mode error')
        p.start()
    finally:
        cleanup_distributed()
