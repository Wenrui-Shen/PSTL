"""PGA input construction for skeleton sequences."""

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import nn

from .gatr import GATr, MLPConfig, SelfAttentionConfig, embed_point, embed_translation


# Pairs follow the PSTL convention: (child, parent), using one-based NTU joint indices.
NTU_BONE_PAIRS: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (2, 21),
    (3, 21),
    (4, 3),
    (5, 21),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 21),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (21, 21),
    (22, 23),
    (23, 8),
    (24, 25),
    (25, 12),
)


class SkeletonPGAInput(nn.Module):
    """Convert PSTL skeleton tensors into GATr multivector inputs.

    Input tensors have shape ``(N, 3, T, V, M)``. Person instances are folded into the batch
    dimension, joint coordinates are embedded as PGA points, and motion and bone displacements are
    embedded as PGA translators. Two feature streams are formed:

    ``joint point + motion translator`` and ``joint point + bone translator``.

    Consecutive frames are then folded into the multivector-channel dimension. With the default
    refinement factor of two, the output shape is ``(N*M, T/2*V, 4, 16)``.
    """

    def __init__(
        self,
        temporal_refinement: int = 2,
        bone_pairs: Sequence[Tuple[int, int]] = NTU_BONE_PAIRS,
    ) -> None:
        super().__init__()
        if temporal_refinement < 1:
            raise ValueError("temporal_refinement must be a positive integer")

        child = torch.tensor([pair[0] - 1 for pair in bone_pairs], dtype=torch.long)
        parent = torch.tensor([pair[1] - 1 for pair in bone_pairs], dtype=torch.long)
        if child.numel() == 0:
            raise ValueError("bone_pairs must not be empty")

        self.temporal_refinement = temporal_refinement
        self.register_buffer("bone_child", child, persistent=False)
        self.register_buffer("bone_parent", parent, persistent=False)

    def forward(
        self,
        skeleton: torch.Tensor,
        ignore_joint: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """Construct GATr inputs from raw joint coordinates.

        Parameters
        ----------
        skeleton : torch.Tensor with shape (N, 3, T, V, M)
            Raw joint coordinates.
        ignore_joint : iterable of int or None
            Zero-based joint indices to mask from all constructed PGA streams.

        Returns
        -------
        multivectors : torch.Tensor with shape
            (N*M, T/temporal_refinement*V, 2*temporal_refinement, 16)
        """
        batch_size, _, num_frames, num_joints, num_people = self._validate_input(skeleton)
        multivectors = self.embed_skeleton(skeleton, ignore_joint)
        multivectors = multivectors.view(
            batch_size * num_people,
            num_frames,
            num_joints,
            2,
            16,
        )

        refined_frames = num_frames // self.temporal_refinement
        multivectors = multivectors.view(
            batch_size * num_people,
            refined_frames,
            self.temporal_refinement,
            num_joints,
            2,
            16,
        )
        # Fold each group of consecutive frames into the MV-channel dimension.
        multivectors = multivectors.permute(0, 1, 3, 2, 4, 5).contiguous()
        multivectors = multivectors.view(
            batch_size * num_people,
            refined_frames * num_joints,
            self.temporal_refinement * 2,
            16,
        )
        return multivectors

    def embed_skeleton(
        self,
        skeleton: torch.Tensor,
        ignore_joint: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """Build the two unrefined PGA streams.

        Returns
        -------
        multivectors : torch.Tensor with shape (N*M, T*V, 2, 16)
            Channels contain ``joint + motion`` and ``joint + bone`` respectively.
        """
        batch_size, channels, num_frames, num_joints, num_people = self._validate_input(skeleton)

        # (N, C, T, V, M) -> (N*M, T, V, C)
        coordinates = skeleton.permute(0, 4, 2, 3, 1).contiguous()
        coordinates = coordinates.view(batch_size * num_people, num_frames, num_joints, channels)

        joint_mv = embed_point(coordinates)
        motion_mv = embed_translation(self._build_motion(coordinates))
        bone_mv = embed_translation(self._build_bone(coordinates))

        # (N*M, T, V, 2, 16): [joint + motion, joint + bone]
        multivectors = torch.stack((joint_mv + motion_mv, joint_mv + bone_mv), dim=-2)
        multivectors = self._mask_joints(multivectors, ignore_joint)
        return multivectors.view(batch_size * num_people, num_frames * num_joints, 2, 16)

    def _validate_input(self, skeleton: torch.Tensor) -> Tuple[int, int, int, int, int]:
        if skeleton.ndim != 5:
            raise ValueError(
                "Skeleton input must have shape (N, C, T, V, M), "
                f"found {tuple(skeleton.shape)}"
            )

        batch_size, channels, num_frames, num_joints, num_people = skeleton.shape
        if channels != 3:
            raise ValueError(f"Skeleton coordinates require C=3, found C={channels}")
        if num_joints != self.bone_child.numel():
            raise ValueError(
                f"Expected {self.bone_child.numel()} joints from bone_pairs, found {num_joints}"
            )
        if num_frames % self.temporal_refinement != 0:
            raise ValueError(
                f"T={num_frames} must be divisible by temporal_refinement="
                f"{self.temporal_refinement}"
            )

        return batch_size, channels, num_frames, num_joints, num_people

    @staticmethod
    def _build_motion(coordinates: torch.Tensor) -> torch.Tensor:
        """Match PSTL motion construction while retaining the original frame count."""
        motion = torch.zeros_like(coordinates)
        motion[:, :-1] = coordinates[:, 1:] - coordinates[:, :-1]
        return motion

    def _build_bone(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Build child-minus-parent bone vectors using the predefined skeleton tree."""
        child = self.bone_child.to(coordinates.device)
        parent = self.bone_parent.to(coordinates.device)
        bone = torch.zeros_like(coordinates)
        bone[:, :, child] = coordinates[:, :, child] - coordinates[:, :, parent]
        return bone

    @staticmethod
    def _mask_joints(
        multivectors: torch.Tensor,
        ignore_joint: Optional[Iterable[int]],
    ) -> torch.Tensor:
        if ignore_joint is None:
            return multivectors

        ignored = sorted(set(ignore_joint))
        if not ignored:
            return multivectors
        num_joints = multivectors.shape[2]
        if ignored[0] < 0 or ignored[-1] >= num_joints:
            raise ValueError(f"ignore_joint contains an index outside [0, {num_joints})")

        joint_mask = torch.ones(
            num_joints, device=multivectors.device, dtype=multivectors.dtype
        )
        joint_mask[ignored] = 0.0
        return multivectors * joint_mask.view(1, 1, num_joints, 1, 1)


class SkeletonGATrEncoder(nn.Module):
    """GATr encoder with the same output contract as the PSTL ST-GCN encoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        hidden_mv_channels: int = 8,
        hidden_s_channels: int = 64,
        num_blocks: int = 6,
        num_heads: int = 4,
        temporal_refinement: int = 2,
        dropout_prob: Optional[float] = None,
        checkpoint_blocks: bool = True,
    ) -> None:
        super().__init__()
        self.input_adapter = SkeletonPGAInput(temporal_refinement=temporal_refinement)
        self.gatr = GATr(
            in_mv_channels=2 * temporal_refinement,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=1,
            out_s_channels=hidden_dim,
            hidden_s_channels=hidden_s_channels,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                multi_query=True,
                pos_encoding=True,
            ),
            mlp=MLPConfig(),
            num_blocks=num_blocks,
            checkpoint=["block"] if checkpoint_blocks else None,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        skeleton: torch.Tensor,
        ignore_joint: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        """Encode ``(N, 3, T, V, M)`` skeletons as ``(N, hidden_dim)`` features."""
        batch_size, _, _, _, num_people = skeleton.shape
        multivectors = self.input_adapter(skeleton, ignore_joint=ignore_joint)
        scalars = torch.zeros(
            *multivectors.shape[:-2],
            1,
            device=multivectors.device,
            dtype=multivectors.dtype,
        )

        _, scalar_outputs = self.gatr(multivectors, scalars=scalars)
        if scalar_outputs is None:
            raise RuntimeError("SkeletonGATrEncoder requires scalar outputs from GATr")

        features = scalar_outputs.mean(dim=1)
        features = features.view(batch_size, num_people, -1).mean(dim=1)
        return features


__all__ = ["NTU_BONE_PAIRS", "SkeletonPGAInput", "SkeletonGATrEncoder"]
