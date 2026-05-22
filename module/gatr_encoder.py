from pathlib import Path
import sys

import torch
import torch.nn as nn


try:
    from gatr import AxialGATr, MLPConfig, SelfAttentionConfig
    from gatr.interface import embed_point
except ModuleNotFoundError:
    gatr_root = Path(__file__).resolve().parents[1] / "geometric-algebra-transformer"
    if gatr_root.exists():
        sys.path.insert(0, str(gatr_root))
    from gatr import AxialGATr, MLPConfig, SelfAttentionConfig
    from gatr.interface import embed_point


class SkeletonGATrEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_frame,
        num_joint,
        num_person,
        in_mv_channels=1,
        out_mv_channels=1,
        hidden_mv_channels=16,
        in_s_channels=256,
        hidden_s_channels=256,
        num_blocks=4,
        num_heads=4,
    ):
        super().__init__()
        self.num_frame = num_frame
        self.num_joint = num_joint
        self.num_person = num_person
        self.in_s_channels = in_s_channels

        self.gatr = AxialGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            attention=SelfAttentionConfig(num_heads=num_heads),
            mlp=MLPConfig(),
            num_blocks=num_blocks,
            pos_encodings=(True, True),
        )

        self.point_scalar_embedding = nn.Sequential(
            nn.Linear(3, in_s_channels),
            nn.LayerNorm(in_s_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_s_channels, in_s_channels),
        )
        self.scalar_projection = nn.Linear(out_mv_channels, hidden_size)

    def forward(self, x, ignore_joint=None):
        n, c, t, v, m = x.shape
        if c != 3:
            raise ValueError("SkeletonGATrEncoder expects 3 coordinate channels.")
        if t > self.num_frame or v != self.num_joint or m != self.num_person:
            raise ValueError(
                "Input skeleton shape does not match configured GATr skeleton dimensions."
            )

        points = x.permute(0, 2, 3, 4, 1).contiguous()
        if ignore_joint:
            remain_joint = sorted(set(range(v)) - set(ignore_joint))
            points = points[:, :, remain_joint, :, :]

        points = points.contiguous().view(n, t, -1, 3)
        multivectors = embed_point(points).unsqueeze(-2)

        scalars = self.point_scalar_embedding(points)

        output_mv, _ = self.gatr(multivectors, scalars=scalars)
        scalar_features = output_mv[..., 0].mean(dim=(1, 2))
        return self.scalar_projection(scalar_features)
