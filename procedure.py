from config import *
from model import *
from dataset import DataSet, Feeder_semi
from logger import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from einops import rearrange, repeat
from math import pi, cos

from module.gcn.st_gcn import Model
from module.gatr_skeleton import SkeletonGATrEncoder


@ex.capture
def build_encoder(encoder_type, in_channels, hidden_channels, hidden_dim, dropout,
                  graph_args, edge_importance_weighting, gatr_out_mv_channels,
                  gatr_in_s_channels, gatr_hidden_mv_channels,
                  gatr_hidden_s_channels, gatr_out_s_channels,
                  gatr_num_blocks, gatr_num_heads,
                  gatr_temporal_refinement, gatr_dropout, gatr_checkpoint_blocks,
                  max_frame, joint_num, person_num):
    if encoder_type == "stgcn":
        return Model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
        )
    if encoder_type == "gatr":
        return SkeletonGATrEncoder(
            out_mv_channels=gatr_out_mv_channels,
            in_s_channels=gatr_in_s_channels,
            hidden_mv_channels=gatr_hidden_mv_channels,
            hidden_s_channels=gatr_hidden_s_channels,
            out_s_channels=gatr_out_s_channels,
            num_blocks=gatr_num_blocks,
            num_heads=gatr_num_heads,
            temporal_refinement=gatr_temporal_refinement,
            num_frames=max_frame,
            num_joints=joint_num,
            num_people=person_num,
            dropout_prob=gatr_dropout,
            checkpoint_blocks=gatr_checkpoint_blocks,
        )
    raise ValueError("encoder_type must be 'stgcn' or 'gatr', found {}".format(encoder_type))


def get_encoder_output_dim(encoder, hidden_size):
    return getattr(encoder, "output_dim", hidden_size)


@ex.capture
def prepare_encoder_input(data, encoder_type):
    if encoder_type == "gatr":
        return data
    return get_stream(data)


def select_encoder_config(encoder_type, override, stgcn_value, gatr_value, name):
    if override is not None:
        return override
    if encoder_type == "stgcn":
        return stgcn_value
    if encoder_type == "gatr":
        return gatr_value
    raise ValueError(
        "encoder_type must be 'stgcn' or 'gatr' when selecting {}, found {}".format(
            name, encoder_type
        )
    )


@ex.capture
def resolve_weight_path(
    encoder_type, weight_path, stgcn_weight_path, gatr_weight_path
):
    return select_encoder_config(
        encoder_type, weight_path, stgcn_weight_path, gatr_weight_path, "weight_path"
    )


@ex.capture
def resolve_log_path(encoder_type, log_path, stgcn_log_path, gatr_log_path):
    return select_encoder_config(
        encoder_type, log_path, stgcn_log_path, gatr_log_path, "log_path"
    )


@ex.capture
def resolve_result_path(
    encoder_type, result_path, stgcn_result_path, gatr_result_path
):
    return select_encoder_config(
        encoder_type, result_path, stgcn_result_path, gatr_result_path, "result_path"
    )

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
setup_seed(1)

class BaseProcessor:

    @ex.capture
    def load_data(self,train_list,train_label,test_list,test_label,batch_size,
                  gatr_batch_size,encoder_type,label_percent):
        self.dataset = dict()
        self.data_loader = dict()
        current_batch_size = gatr_batch_size if encoder_type == "gatr" else batch_size

        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)
        # self.dataset['semi'] = Feeder_semi(train_list, train_label, label_percent)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=current_batch_size,
            num_workers=32,
            shuffle=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=current_batch_size,
            num_workers=32,
            shuffle=False)
        
        # self.data_loader['semi'] = torch.utils.data.DataLoader(
        #     dataset=self.dataset['semi'],
        #     batch_size=batch_size,
        #     num_workers=32,
        #     shuffle=True)
        
    def load_weights(self, model=None, weight_path=None):
        if weight_path:
            pretrained_dict = torch.load(weight_path)
            model.load_state_dict(pretrained_dict)

    @ex.capture
    def initialize(self, train_mode, resume_path):
        self.weight_path = resolve_weight_path()
        self.log_path = resolve_log_path()
        self.result_path = resolve_result_path()
        self.load_data()
        self.load_model()
        self.load_optim()
        append_log = "pretrain" in train_mode and resume_path is not None
        self.log = Log(log_path=self.log_path, append=append_log)
    
    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
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
    def load_model(self,train_mode,hidden_size):
        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        feature_size = get_encoder_output_dim(self.encoder, hidden_size)
        self.classifier = Linear(hidden_size=feature_size).cuda()
        self.load_weights(self.encoder, self.weight_path)
    
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
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = prepare_encoder_input(data)
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
    def test_epoch(self, epoch, label_path, save_lp):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = self.result_path + str(epoch) + '_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = prepare_encoder_input(data)

            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)
                label_list.append(label)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_lp:
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)

    def save_model(self):
        
        pass
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.epoch = epoch
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class SemiProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,hidden_size):
        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        feature_size = get_encoder_output_dim(self.encoder, hidden_size)
        self.classifier = Linear(hidden_size=feature_size).cuda()
        self.load_weights(self.encoder, self.weight_path)
    
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
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = prepare_encoder_input(data)
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
    def test_epoch(self, epoch, label_path, save_semi=True):
        self.encoder.eval()
        self.classifier.eval()

        result_list = []
        label_list = []
        r_path = self.result_path + str(epoch) + '_semi10_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = prepare_encoder_input(data)
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)
                label_list.append(label)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/semi_test/cls_acc", acc.item())
            self.log.update_batch("log/semi_test/cls_loss", loss.item())

        if save_semi:
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)
    
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class FTProcessor(BaseProcessor):

    @ex.capture
    def load_model(self,train_mode,hidden_size):
        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        feature_size = get_encoder_output_dim(self.encoder, hidden_size)
        self.classifier = Linear(hidden_size=feature_size).cuda()
        self.load_weights(self.encoder, self.weight_path)
    
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
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = prepare_encoder_input(data)
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
    def test_epoch(self, epoch, label_path, save_finetune):
        self.encoder.eval()
        self.classifier.eval()
        result_list = []
        label_list = []
        r_path = self.result_path + str(epoch) + '_finetune_result.pkl'

        loader = self.data_loader['test']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            data = prepare_encoder_input(data)
            with torch.no_grad():
                Z = self.encoder(data)
                predict = self.classifier(Z)
                result_list.append(predict)
                label_list.append(label)

            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

        if save_finetune:
            torch.save(result_list, r_path)
            torch.save(label_list, label_path)
    @ex.capture
    def optimize(self,lp_epoch):
        for epoch in range(lp_epoch):
            print("epoch:",epoch)
            self.train_epoch()
            self.test_epoch(epoch)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log.update_epoch(epoch,lp_epoch,lr=lr)

class BTProcessor(BaseProcessor):
    
    @ex.capture
    def load_model(self, hidden_size, encoder_type):
        self.encoder = build_encoder()
        self.encoder = self.encoder.cuda()
        feature_size = get_encoder_output_dim(self.encoder, hidden_size)
        if encoder_type == "gatr":
            self.btwins_head = GATrBTwins(hidden_size=feature_size).cuda()
            self.boundary_head = GATrBoundaryHead(hidden_size=feature_size).cuda()
        else:
            self.btwins_head = BTwins(hidden_size=feature_size).cuda()
            self.boundary_head = None

    @ex.capture
    def load_optim(self, pretrain_lr, pretrain_epoch, weight_decay, resume_path,
                   encoder_type):
        parameter_groups = [
            {'params': self.encoder.parameters()},
            {'params': self.btwins_head.parameters()},
        ]
        if self.boundary_head is not None:
            parameter_groups.append({'params': self.boundary_head.parameters()})
        self.optimizer = torch.optim.Adam(
            parameter_groups,
            weight_decay=weight_decay,
            lr=pretrain_lr,
        )
        self.start_epoch = 0
        if resume_path:
            self.load_checkpoint(resume_path, encoder_type)

    def load_checkpoint(self, resume_path, encoder_type):
        checkpoint = torch.load(resume_path)
        required_keys = {"epoch", "encoder", "btwins_head", "optimizer"}
        if encoder_type == "gatr":
            required_keys.add("boundary_head")
        if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
            raise ValueError(
                "resume_path must point to a pretraining checkpoint containing "
                "epoch, encoder, btwins_head and optimizer"
            )

        saved_encoder_type = checkpoint.get("encoder_type")
        if saved_encoder_type is not None and saved_encoder_type != encoder_type:
            raise ValueError(
                "Checkpoint encoder_type is {}, but current encoder_type is {}".format(
                    saved_encoder_type, encoder_type
                )
            )

        self.encoder.load_state_dict(checkpoint["encoder"])
        self.btwins_head.load_state_dict(checkpoint["btwins_head"])
        if self.boundary_head is not None:
            self.boundary_head.load_state_dict(checkpoint["boundary_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"] + 1

        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if "numpy_rng_state" in checkpoint:
            numpy_rng_state = checkpoint["numpy_rng_state"]
            numpy_state = numpy_rng_state["state"]
            if torch.is_tensor(numpy_state):
                numpy_state = numpy_state.cpu().numpy()
            np.random.set_state(
                (
                    numpy_rng_state["bit_generator"],
                    np.asarray(numpy_state, dtype=np.uint32),
                    numpy_rng_state["position"],
                    numpy_rng_state["has_gauss"],
                    numpy_rng_state["cached_gaussian"],
                )
            )
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

        print(
            "Resumed pretraining from epoch {} using {}".format(
                self.start_epoch, resume_path
            )
        )

    def btwins_batch(self, feat1, feat2, mode):
        BTloss = self.btwins_head(feat1, feat2)
        BTloss = torch.mean(BTloss)
        self.log.update_batch("log/pretrain/"+mode+"_bt_loss", BTloss.item())
        return BTloss

    def btwins_multiview(self, features, mode):
        projected = [self.btwins_head.project(item) for item in features]
        pair_losses = []
        for first in range(len(projected)):
            for second in range(first + 1, len(projected)):
                pair_losses.append(
                    self.btwins_head.loss_from_projected(
                        projected[first],
                        projected[second],
                    )
                )
        loss = torch.stack(pair_losses).mean()
        self.log.update_batch(
            "log/pretrain/" + mode + "_bt_loss",
            loss.item(),
        )
        return loss

    @ex.capture
    def boundary_batch(
        self,
        equivariant_features,
        non_equivariant_features,
        gatr_boundary_beta,
        gatr_boundary_margin,
        gatr_boundary_width,
        gatr_boundary_variance_gamma,
    ):
        if gatr_boundary_width < 0:
            raise ValueError("gatr_boundary_width must be non-negative")

        projected_equivariant = []
        normalized_equivariant = []
        for features in equivariant_features:
            projected, normalized = self.boundary_head(features)
            projected_equivariant.append(projected)
            normalized_equivariant.append(normalized.detach())

        projected_equivariant = torch.stack(projected_equivariant, dim=1)
        normalized_equivariant = torch.stack(normalized_equivariant, dim=1)
        _, normalized_non_equivariant = self.boundary_head(
            non_equivariant_features
        )

        center = F.normalize(normalized_equivariant.mean(dim=1), dim=-1)
        equivariant_distances = 1.0 - torch.einsum(
            "nkd,nd->nk",
            normalized_equivariant,
            center,
        )
        radius = (
            equivariant_distances.mean(dim=1)
            + gatr_boundary_beta
            * equivariant_distances.std(dim=1, unbiased=False)
        )

        non_equivariant_distance = 1.0 - (
            normalized_non_equivariant * center.detach()
        ).sum(dim=-1)
        lower_boundary = (
            radius.detach() + gatr_boundary_margin
        ).clamp(max=2.0)
        upper_boundary = (
            lower_boundary + gatr_boundary_width
        ).clamp(max=2.0)

        lower_violation = F.relu(
            lower_boundary - non_equivariant_distance
        )
        upper_violation = F.relu(
            non_equivariant_distance - upper_boundary
        )
        boundary_loss = (
            lower_violation.square() + upper_violation.square()
        ).mean()
        boundary_hit_rate = (
            (non_equivariant_distance >= lower_boundary)
            & (non_equivariant_distance <= upper_boundary)
        ).float().mean()

        flat_equivariant = projected_equivariant.flatten(0, 1)
        feature_std = torch.sqrt(
            flat_equivariant.var(dim=0, unbiased=False) + 1e-4
        )
        variance_loss = F.relu(
            gatr_boundary_variance_gamma - feature_std
        ).mean()

        self.log.update_batch(
            "log/pretrain/boundary_loss",
            boundary_loss.item(),
        )
        self.log.update_batch(
            "log/pretrain/boundary_variance_loss",
            variance_loss.item(),
        )
        self.log.update_batch(
            "log/pretrain/equivariant_radius",
            radius.mean().item(),
        )
        self.log.update_batch(
            "log/pretrain/non_equivariant_distance",
            non_equivariant_distance.mean().item(),
        )
        self.log.update_batch(
            "log/pretrain/boundary_lower",
            lower_boundary.mean().item(),
        )
        self.log.update_batch(
            "log/pretrain/boundary_upper",
            upper_boundary.mean().item(),
        )
        self.log.update_batch(
            "log/pretrain/boundary_hit_rate",
            boundary_hit_rate.item(),
        )
        return boundary_loss, variance_loss

    @ex.capture
    def train_epoch(self, epoch, pretrain_epoch, pretrain_lr, encoder_type,
                    gatr_translation_range, gatr_y_rotation_degrees,
                    gatr_reflection_prob, gatr_num_equivariant_views,
                    gatr_boundary_loss_weight,
                    gatr_boundary_variance_weight,
                    gatr_non_eq_shear_prob, gatr_non_eq_shear_amplitude,
                    gatr_non_eq_noise_prob, gatr_non_eq_noise_std,
                    gatr_non_eq_blur_prob, gatr_non_eq_blur_kernel,
                    gatr_non_eq_blur_sigma_min, gatr_non_eq_blur_sigma_max,
                    gatr_non_eq_axis_mask_prob):
        self.encoder.train()
        self.btwins_head.train()
        if self.boundary_head is not None:
            self.boundary_head.train()

        loader = self.data_loader['train']
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=pretrain_epoch, lr_max=pretrain_lr)
        
        for data, label in tqdm(loader):
            # load data
            n,c,t,v,m = data.shape
            data = data.type(torch.FloatTensor).cuda()
            data = prepare_encoder_input(data)

            if encoder_type == "gatr":
                if gatr_num_equivariant_views < 2:
                    raise ValueError(
                        "gatr_num_equivariant_views must be at least 2"
                    )

                equivariant_features = []
                for _ in range(gatr_num_equivariant_views):
                    equivariant_x = gatr_random_translation(
                        data,
                        translation_range=gatr_translation_range,
                    )
                    equivariant_x = gatr_random_y_rotation(
                        equivariant_x,
                        y_rotation_degrees=gatr_y_rotation_degrees,
                    )
                    equivariant_x = gatr_random_reflection(
                        equivariant_x,
                        reflection_prob=gatr_reflection_prob,
                    )
                    equivariant_features.append(self.encoder(equivariant_x))

                non_equivariant_x = gatr_non_equivariant_augmentation(
                    data,
                    shear_prob=gatr_non_eq_shear_prob,
                    shear_amplitude=gatr_non_eq_shear_amplitude,
                    noise_prob=gatr_non_eq_noise_prob,
                    noise_std=gatr_non_eq_noise_std,
                    blur_prob=gatr_non_eq_blur_prob,
                    blur_kernel=gatr_non_eq_blur_kernel,
                    blur_sigma_min=gatr_non_eq_blur_sigma_min,
                    blur_sigma_max=gatr_non_eq_blur_sigma_max,
                    axis_mask_prob=gatr_non_eq_axis_mask_prob,
                )
                non_equivariant_features = self.encoder(non_equivariant_x)

                loss_bt = self.btwins_multiview(
                    equivariant_features,
                    mode="equivariant",
                )
                loss_boundary, loss_variance = self.boundary_batch(
                    equivariant_features,
                    non_equivariant_features,
                )
                loss = (
                    loss_bt
                    + gatr_boundary_loss_weight * loss_boundary
                    + gatr_boundary_variance_weight * loss_variance
                )
            else:
                # Preserve the original PSTL streams and pretraining tasks for ST-GCN.
                ignore_joint = central_spacial_mask()

                input1 = shear(crop(data))
                input1 = random_rotate(input1)
                input1 = random_spatial_flip(input1)
                feat1 = self.encoder(input1)

                input2 = shear(crop(data))
                input2 = random_rotate(input2)
                input2 = random_spatial_flip(input2)
                input2 = motion_att_temp_mask(input2)
                feat2 = self.encoder(input2)

                input3 = shear(crop(data))
                input3 = random_rotate(input3)
                input3 = random_spatial_flip(input3)
                feat3 = self.encoder(input3, ignore_joint)

                loss_bt1 = self.btwins_batch(feat1, feat2, mode='temp_mask')
                loss_bt2 = self.btwins_batch(feat1, feat3, mode='joint_mask')
                loss = loss_bt1 + loss_bt2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    @ex.capture
    def save_model(self, epoch):
        os.makedirs(os.path.dirname(self.weight_path) or ".", exist_ok=True)
        torch.save(self.encoder.state_dict(), self.weight_path)

    @ex.capture
    def save_checkpoint(self, epoch, encoder_type):
        checkpoint_path = self.weight_path
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        numpy_rng_state = np.random.get_state()
        checkpoint = {
                "epoch": epoch,
                "encoder_type": encoder_type,
                "encoder": self.encoder.state_dict(),
                "btwins_head": self.btwins_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "numpy_rng_state": {
                    "bit_generator": numpy_rng_state[0],
                    # PyTorch does not support tensors backed by NumPy uint32 arrays.
                    # MT19937 values fit exactly in int64 and are cast back on restore.
                    "state": torch.from_numpy(numpy_rng_state[1].astype(np.int64)),
                    "position": numpy_rng_state[2],
                    "has_gauss": numpy_rng_state[3],
                    "cached_gaussian": numpy_rng_state[4],
                },
                "python_rng_state": random.getstate(),
        }
        if self.boundary_head is not None:
            checkpoint["boundary_head"] = self.boundary_head.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print("Saved pretraining checkpoint to {}".format(checkpoint_path))
        
    @ex.capture
    def optimize(self, pretrain_epoch, checkpoint_interval):
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")

        for epoch in range(self.start_epoch, pretrain_epoch):
            print("epoch:",epoch)
            self.epoch = epoch
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


# %%
@ex.automain
def main(train_mode):
    if "pretrain" in train_mode:
        p = BTProcessor()
    elif "lp" in train_mode:
        p = RecognitionProcessor()
    elif "finetune" in train_mode:
        p = FTProcessor()
    elif "semi" in train_mode:
        p = SemiProcessor()
    else:
        print('train_mode error')
    p.start()
