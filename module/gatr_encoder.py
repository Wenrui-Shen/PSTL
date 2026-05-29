from pathlib import Path
import sys

import torch
import torch.nn as nn


try:
    from gatr import GATr, MLPConfig, SelfAttentionConfig
    from gatr.interface import embed_point
except ModuleNotFoundError:
    gatr_root = Path(__file__).resolve().parents[1] / "geometric-algebra-transformer"
    if gatr_root.exists():
        sys.path.insert(0, str(gatr_root))
    from gatr import GATr, MLPConfig, SelfAttentionConfig
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
        checkpoint_blocks=False,
    ):
        super().__init__()
        self.num_frame = num_frame
        self.num_joint = num_joint
        self.num_person = num_person

        self.spatial_tokens = num_joint * num_person
        self.temporal_tokens = num_frame

        self.time_gatr = GATr(
            in_mv_channels=self.spatial_tokens,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=None,
            attention=SelfAttentionConfig(num_heads=num_heads),
            mlp=MLPConfig(),
            num_blocks=num_blocks,
            checkpoint_blocks=checkpoint_blocks,
        )

        self.space_gatr = GATr(
            in_mv_channels=self.temporal_tokens,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=None,
            attention=SelfAttentionConfig(num_heads=num_heads),
            mlp=MLPConfig(),
            num_blocks=num_blocks,
            checkpoint_blocks=checkpoint_blocks,
        )

        mv_feature_size = out_mv_channels * 16
        self.time_projection = nn.Sequential(
            nn.LayerNorm(mv_feature_size),
            nn.Linear(mv_feature_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.space_projection = nn.Sequential(
            nn.LayerNorm(mv_feature_size),
            nn.Linear(mv_feature_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.instance_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x, ignore_joint=None):
        n, c, t, v, m = x.shape
        if c != 3:
            raise ValueError("SkeletonGATrEncoder expects 3 coordinate channels.")
        if t > self.num_frame or v != self.num_joint or m != self.num_person:
            raise ValueError(
                "Input skeleton shape does not match configured GATr skeleton dimensions."
            )

        points = x.permute(0, 2, 3, 4, 1).contiguous()
        point_multivectors = embed_point(points)
        if ignore_joint:
            point_multivectors = point_multivectors.clone()
            point_multivectors[:, :, ignore_joint, :, :] = 0

        multivectors = point_multivectors.contiguous().view(n, t, self.spatial_tokens, 16)

        time_output_mv, _ = self.time_gatr(multivectors, scalars=None)
        time_token_features = time_output_mv.flatten(2)
        time_features = self.time_projection(time_token_features).mean(dim=1)

        space_multivectors = multivectors.transpose(1, 2).contiguous()
        if t < self.temporal_tokens:
            padding = space_multivectors.new_zeros(
                n,
                self.spatial_tokens,
                self.temporal_tokens - t,
                16,
            )
            space_multivectors = torch.cat([space_multivectors, padding], dim=2)

        space_output_mv, _ = self.space_gatr(space_multivectors, scalars=None)
        space_token_features = space_output_mv.flatten(2)
        space_features = self.space_projection(space_token_features).mean(dim=1)

        instance_features = torch.cat([time_features, space_features], dim=1)
        return self.instance_projection(instance_features)
