from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

from math import sin, cos
from einops import rearrange, repeat

def init_weights(m):
    class_name=m.__class__.__name__

    if "Conv2d" in class_name or "Linear" in class_name:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    if class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Linear(nn.Module):
    @ex.capture
    def __init__(self, hidden_size, dataset): 
        super(Linear, self).__init__()
        if "ntu60" in dataset:
            label_num = 60
        elif "ntu120" in dataset:
            label_num = 120
        elif "pku" in dataset:
            label_num = 51
        else:
            raise ValueError
        self.classifier = nn.Linear(hidden_size, label_num)
        self.apply(init_weights)

    def forward(self, X):
        X = self.classifier(X)
        return X

class BTwins(nn.Module):

    @ex.capture
    def __init__(self, hidden_size, lambd, pj_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
        )
        self.bn = nn.BatchNorm1d(pj_size, affine=False)
        self.lambd = lambd

    def forward(self, feat1, feat2):
        
        feat1 = self.projector(feat1)
        feat2 = self.projector(feat2)
        feat1_norm = self.bn(feat1)
        feat2_norm = self.bn(feat2)

        N, D = feat1_norm.shape
        c = (feat1_norm.T @ feat2_norm).div_(N)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        BTloss = on_diag + self.lambd * off_diag

        return BTloss 

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class GATrBTwins(nn.Module):

    @ex.capture
    def __init__(self, hidden_size, lambd, gatr_pj_size):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, gatr_pj_size, bias=False),
            nn.BatchNorm1d(gatr_pj_size),
            nn.ReLU(True),
            nn.Linear(gatr_pj_size, gatr_pj_size, bias=False),
            nn.BatchNorm1d(gatr_pj_size),
            nn.ReLU(True),
            nn.Linear(gatr_pj_size, gatr_pj_size, bias=False),
        )
        self.bn = nn.BatchNorm1d(gatr_pj_size, affine=False)
        self.lambd = lambd

    def project(self, features):
        return self.bn(self.projector(features))

    def loss_from_projected(self, feat1, feat2):
        batch_size = feat1.shape[0]
        correlation = (feat1.T @ feat2).div_(batch_size)
        on_diag = torch.diagonal(correlation).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(correlation).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    def forward(self, feat1, feat2):
        return self.loss_from_projected(
            self.project(feat1),
            self.project(feat2),
        )

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class GATrContrastiveHead(nn.Module):

    @ex.capture
    def __init__(
        self,
        hidden_size,
        gatr_contrastive_hidden_size,
        gatr_contrastive_size,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, gatr_contrastive_hidden_size, bias=False),
            nn.LayerNorm(gatr_contrastive_hidden_size),
            nn.ReLU(True),
            nn.Linear(gatr_contrastive_hidden_size, gatr_contrastive_size, bias=False),
        )

    def forward(self, features):
        projected = self.projector(features)
        normalized = F.normalize(projected, dim=-1)
        return projected, normalized


GATrBoundaryHead = GATrContrastiveHead
    
@ex.capture 
def get_stream(data, view):
    N, C, T, V, M = data.shape

    if view == 'joint':
        pass

    elif view == 'motion':
        motion = torch.zeros_like(data)
        motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

        data = motion

    elif view == 'bone':
        Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        bone = torch.zeros_like(data)

        for v1, v2 in Bone:
            bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

        data = bone
    
    else:

        return None

    return data

@ex.capture
def shear(input_data, shear_amp):
    # n c t v m
    temp = input_data.clone()
    amp = shear_amp
    Shear       = np.array([
                    [1, random.uniform(-amp, amp), 	random.uniform(-amp, amp)],
                    [random.uniform(-amp, amp), 1, 	random.uniform(-amp, amp)],
                    [random.uniform(-amp, amp), 	random.uniform(-amp, amp),1]
                    ])
    Shear = torch.Tensor(Shear).cuda()
    output =  torch.einsum('n c t v m, c d -> n d t v m',[temp,Shear])

    return output
    
def reverse(data,p=0.5):

    N,C,T,V,M = data.shape
    temp = data.clone()

    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return temp[:,:, time_range_reverse, :, :]
    else:
        return temp
        
@ex.capture 
def crop(data, temperal_padding_ratio=6):
    input_data = data.clone()
    N, C, T, V, M = input_data.shape
    #padding
    padding_len = T // temperal_padding_ratio
    frame_start = torch.randint(0, padding_len * 2 + 1,(1,))
    first_clip = torch.flip(input_data[:,:,:padding_len],dims=[2])
    second_clip = input_data
    thrid_clip = torch.flip(input_data[:,:,-padding_len:],dims=[2])
    out = torch.cat([first_clip,second_clip,thrid_clip],dim=2)
    out = out[:, :, frame_start:frame_start + T]
    
    return out

def random_rotate(data):
    def rotate(seq, axis, angle):
        # x
        if axis == 0:
            R = np.array([[1, 0, 0],
                              [0, cos(angle), sin(angle)],
                              [0, -sin(angle), cos(angle)]])
        # y
        if axis == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                              [0, 1, 0],
                              [sin(angle), 0, cos(angle)]])

        # z
        if axis == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                              [-sin(angle), cos(angle), 0],
                              [0, 0, 1]])
        R = R.T
        R = torch.Tensor(R).cuda()
        output =  torch.einsum('n c t v m, c d -> n d t v m',[seq,R])
        return output

    # n c t v m
    new_seq = data.clone()
    total_axis = [0, 1, 2]
    main_axis = random.randint(0, 2)
    for axis in total_axis:
        if axis == main_axis:
            rotate_angle = random.uniform(0, 30)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)
        else:
            rotate_angle = random.uniform(0, 1)
            rotate_angle = math.radians(rotate_angle)
            new_seq = rotate(new_seq, axis, rotate_angle)

    return new_seq


def gatr_random_translation(data, translation_range=0.5):
    """Translate each skeleton sequence without changing zero padding."""
    if data.ndim != 5 or data.shape[1] != 3:
        raise ValueError(
            "Skeleton input must have shape (N, 3, T, V, M), "
            f"found {tuple(data.shape)}"
        )
    if translation_range < 0:
        raise ValueError("translation_range must be non-negative")

    output = data.clone()
    batch_size = data.shape[0]
    translation = torch.empty(
        batch_size, 3, device=data.device, dtype=data.dtype
    ).uniform_(-translation_range, translation_range)
    valid_person_frames = data.abs().sum(dim=(1, 3), keepdim=True).ne(0)
    return output + translation[:, :, None, None, None] * valid_person_frames


def gatr_random_y_rotation(data, y_rotation_degrees=30.0):
    """Rotate each skeleton sequence around the body y axis."""
    if y_rotation_degrees < 0:
        raise ValueError("y_rotation_degrees must be non-negative")

    batch_size = data.shape[0]
    angles = torch.empty(
        batch_size, device=data.device, dtype=data.dtype
    ).uniform_(-y_rotation_degrees, y_rotation_degrees)
    angles = angles * (math.pi / 180.0)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)

    rotation = torch.zeros(
        batch_size, 3, 3, device=data.device, dtype=data.dtype
    )
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 2] = -sine
    rotation[:, 1, 1] = 1.0
    rotation[:, 2, 0] = sine
    rotation[:, 2, 2] = cosine
    return torch.einsum("nij,njtvp->nitvp", rotation, data)


def gatr_random_reflection(data, reflection_prob=0.5):
    """Reflect each skeleton sequence across the yz plane."""
    if not 0.0 <= reflection_prob <= 1.0:
        raise ValueError("reflection_prob must be in [0, 1]")

    batch_size = data.shape[0]
    reflected = torch.rand(batch_size, device=data.device) < reflection_prob
    reflection = torch.ones(
        batch_size, 3, device=data.device, dtype=data.dtype
    )
    reflection[reflected, 0] = -1.0
    return data * reflection[:, :, None, None, None]


def _gatr_temporal_gaussian_blur(data, kernel_size, sigma):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("Gaussian blur kernel size must be a positive odd number")
    if sigma <= 0:
        raise ValueError("Gaussian blur sigma must be positive")

    radius = kernel_size // 2
    positions = torch.arange(
        -radius,
        radius + 1,
        device=data.device,
        dtype=data.dtype,
    )
    kernel = torch.exp(-(positions ** 2) / (2.0 * sigma ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size)

    batch_size, channels, frames, joints, people = data.shape
    sequences = data.permute(0, 1, 3, 4, 2).reshape(-1, 1, frames)
    blurred = F.conv1d(sequences, kernel, padding=radius)
    return blurred.reshape(
        batch_size, channels, joints, people, frames
    ).permute(0, 1, 4, 2, 3)


def _gatr_random_shear(data, amplitude):
    if amplitude < 0:
        raise ValueError("Shear amplitude must be non-negative")

    batch_size = data.shape[0]
    shear = torch.eye(3, device=data.device, dtype=data.dtype)
    shear = shear.unsqueeze(0).repeat(batch_size, 1, 1)
    random_values = torch.empty(
        batch_size, 3, 3, device=data.device, dtype=data.dtype
    ).uniform_(-amplitude, amplitude)
    off_diagonal = ~torch.eye(3, device=data.device, dtype=torch.bool)
    shear[:, off_diagonal] = random_values[:, off_diagonal]
    return torch.einsum("nij,njtvp->nitvp", shear, data)


def gatr_non_equivariant_augmentation(
    data,
    shear_prob=0.8,
    shear_amplitude=0.2,
    noise_prob=0.8,
    noise_std=0.02,
    blur_prob=0.5,
    blur_kernel=15,
    blur_sigma_min=0.1,
    blur_sigma_max=2.0,
    axis_mask_prob=0.3,
):
    """Apply mild non-equivariant corruptions adapted from feeder/ntu_feeder.py."""
    if data.ndim != 5 or data.shape[1] != 3:
        raise ValueError(
            "Skeleton input must have shape (N, 3, T, V, M), "
            f"found {tuple(data.shape)}"
        )
    for name, probability in (
        ("shear_prob", shear_prob),
        ("noise_prob", noise_prob),
        ("blur_prob", blur_prob),
        ("axis_mask_prob", axis_mask_prob),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if shear_amplitude < 0:
        raise ValueError("shear_amplitude must be non-negative")
    if blur_sigma_min <= 0 or blur_sigma_max < blur_sigma_min:
        raise ValueError("Invalid Gaussian blur sigma range")

    output = data.clone()
    batch_size = data.shape[0]
    valid_positions = data.abs().sum(dim=1, keepdim=True).ne(0)
    applied = torch.zeros(batch_size, device=data.device, dtype=torch.bool)

    shear_mask = torch.rand(batch_size, device=data.device) < shear_prob
    if shear_mask.any() and shear_amplitude > 0:
        sheared = _gatr_random_shear(output, shear_amplitude)
        output = torch.where(
            shear_mask[:, None, None, None, None],
            sheared,
            output,
        )
        applied |= shear_mask

    noise_mask = torch.rand(batch_size, device=data.device) < noise_prob
    if noise_mask.any() and noise_std > 0:
        noise = torch.randn_like(output) * noise_std
        output = output + (
            noise
            * noise_mask[:, None, None, None, None]
            * valid_positions
        )
        applied |= noise_mask

    blur_mask = torch.rand(batch_size, device=data.device) < blur_prob
    if blur_mask.any():
        sigma = torch.empty((), device=data.device).uniform_(
            blur_sigma_min, blur_sigma_max
        ).item()
        blurred = _gatr_temporal_gaussian_blur(output, blur_kernel, sigma)
        output = torch.where(
            blur_mask[:, None, None, None, None],
            blurred,
            output,
        )
        output = output * valid_positions
        applied |= blur_mask

    axis_mask = torch.rand(batch_size, device=data.device) < axis_mask_prob
    if axis_mask.any():
        axes = torch.randint(0, 3, (batch_size,), device=data.device)
        keep = torch.ones(
            batch_size, 3, device=data.device, dtype=data.dtype
        )
        keep[axis_mask, axes[axis_mask]] = 0.0
        output = output * keep[:, :, None, None, None]
        applied |= axis_mask

    # Every sample must include at least one corruption. A small shear is the
    # fallback when all independently sampled corruptions were skipped.
    fallback = ~applied
    if fallback.any():
        fallback_amplitude = (
            shear_amplitude if shear_amplitude > 0 else 0.05
        )
        sheared = _gatr_random_shear(output, fallback_amplitude)
        output = torch.where(
            fallback[:, None, None, None, None],
            sheared,
            output,
        )

    return output * valid_positions


@ex.capture
def get_ignore_joint(mask_joint):

    ignore_joint = random.sample(range(25), mask_joint)
    return ignore_joint

@ex.capture
def get_ignore_part(mask_part):

    left_hand = [8,9,10,11,23,24]
    right_hand = [4,5,6,7,21,22]
    left_leg = [16,17,18,19]
    right_leg = [12,13,14,15]
    body = [0,1,2,3,20]
    all_joint = [left_hand, right_hand, left_leg, right_leg, body]
    part = random.sample(range(5), mask_part)
    ignore_joint = []
    for i in part:
        ignore_joint += all_joint[i]

    return ignore_joint

def gaus_noise(data, mean= 0, std = 0.01):
    temp = data.clone()
    n, c, t, v, m = temp.shape
    noise = np.random.normal(mean, std, size=(n, c, t, v, m))
    noise = torch.Tensor(noise).cuda()

    return temp + noise

def gaus_filter(data):
    temp = data.clone()
    g = GaussianBlurConv(3).cuda()
    return g(temp)

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel = 15, sigma = [0.1, 2]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):
        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        kernel =  kernel.float()
        kernel = kernel.repeat(self.channels, 1, 1, 1) # (3,1,1,5)
        kernel = kernel.cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.weight = self.weight.cuda()

        prob = np.random.random_sample()
        if prob < 0.5:
            #x = x.permute(3,0,2,1) # M,C,V,T
            x = rearrange(x, 'n c t v m -> (n m) c v t')
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            #x = x.permute(1,-1,-2, 0) #C,T,V,M
            x = rearrange(x, '(n m) c v t -> n c t v m', m = 2)

        return x

@ex.capture
def temporal_cropresize(input_data,max_frame,output_size,l_ratio=[0.1,1]):

    num_of_frames = max_frame
    
    n, c, t, v, m = input_data.shape
    min_crop_length = 64
    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)
    start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:, :,start:start+temporal_crop_length, :, :]
    temporal_context = rearrange(temporal_context,'n c t v m -> n (c v m) t')
    temporal_context=temporal_context[: , :, :,None]
    temporal_context= F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear',align_corners=False)
    temporal_context = temporal_context.squeeze(dim=-1)
    temporal_context = rearrange(temporal_context,'n (c v m) t -> n c t v m',c=c,v=v,m=m)
    return temporal_context

def random_spatial_flip(data, p=0.5):
    temp = data.clone()
    order = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 
    17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
    if random.random() < p:
        temp = temp[:, :, :, order, :]

    return temp

def random_time_flip(temp, p=0.5):
    # temp = data.clone()
    T = temp.shape[2]
    if random.random() < p:
        time_range_order = [i for i in range(T)]
        time_range_reverse = list(reversed(time_range_order))
        return temp[:,:, time_range_reverse, :, :]
    else:
        return temp

@ex.capture
def motion_att_temp_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)
    temp_resample = temp.gather(2,temp_list)

    ## random temp mask
    random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    random_frame.sort()
    output = temp_resample[:, :, random_frame, :, :]

    return output

@ex.capture
def central_spacial_mask(mask_joint):

    # Degree Centrality
    degree_centrality = [3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 
                        2, 2, 2, 1, 2, 2, 2, 1, 4, 1, 2, 1, 2]
    all_joint = []
    for i in range(25):
        all_joint += [i]*degree_centrality[i]

    ignore_joint = random.sample(all_joint, mask_joint)

    return ignore_joint


def semi_mask(mask_num):

    p = random.random()
    if p<0.5:
        ignore_joint = central_spacial_mask(mask_num)
    else:
        ignore_joint = []
    
    return ignore_joint
