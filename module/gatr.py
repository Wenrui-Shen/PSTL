# Copyright (c) 2023-2025 Qualcomm Technologies, Inc.
# All rights reserved.
"""Self-contained Geometric Algebra Transformer implementation.

This file is extracted from the bundled geometric-algebra-transformer source tree so that GATr can
be imported from ``module.gatr`` without keeping or installing that tree. The model architecture,
parameterization, public API, and runtime dependencies follow the upstream implementation. The
only portability change is embedding the two file-backed algebra kernels directly in this module.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial, wraps
from itertools import chain, product
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union
from warnings import warn

import numpy as np
import opt_einsum
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
from torch.utils.checkpoint import checkpoint as checkpoint_
from xformers.ops import AttentionBias, memory_efficient_attention


def _einsum_with_path(equation: str, *operands: torch.Tensor, path: List[int]) -> torch.Tensor:
    """Computes einsum with a given contraction path."""
    return torch._VF.einsum(equation, operands, path=path)  # type: ignore[attr-defined]


def _einsum_with_path_ignored(equation: str, *operands: torch.Tensor, **kwargs: Any):
    """Calls torch.einsum whilst dropping all kwargs."""
    return torch.einsum(equation, *operands)


def _cached_einsum(equation: str, *operands: torch.Tensor) -> torch.Tensor:
    """Computes einsum whilst caching the optimal contraction path."""
    op_shape = tuple(op.shape for op in operands)
    path = _get_cached_path_for_equation_and_shapes(equation=equation, op_shape=op_shape)
    return _einsum_with_path(equation, *operands, path=path)


@functools.lru_cache(maxsize=None)
def _get_cached_path_for_equation_and_shapes(
    equation: str, op_shape: Sequence[torch.Tensor]
) -> List[int]:
    """Provides shape-based caching of the optimal contraction path."""
    tupled_path = opt_einsum.contract_path(equation, *op_shape, optimize="optimal", shapes=True)[0]
    return [item for pair in tupled_path for item in pair]


class gatr_cache(dict):
    """Serves as a torch.compile-compatible replacement for functools.cache()."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def __missing__(self, item: Any) -> Any:
        tensor = self.fn(*item)
        self[item] = tensor
        return tensor

    def __call__(self, *args: Any) -> Any:
        return self[args]


_gatr_einsum = _cached_einsum
_gatr_einsum_with_path = _einsum_with_path


def gatr_einsum(equation: str, *operands: torch.Tensor):
    """Computes torch.einsum with contraction path caching if enabled."""
    return _gatr_einsum(equation, *operands)


def gatr_einsum_with_path(equation: str, *operands: torch.Tensor, path: List[int]):
    """Computes einsum with a given contraction path."""
    return _gatr_einsum_with_path(equation, *operands, path=path)


def enable_cached_einsum(flag: bool) -> None:
    """Selects whether to use cached contraction paths in einsum computations."""
    global _gatr_einsum
    global _gatr_einsum_with_path
    if flag:
        _gatr_einsum = _cached_einsum
        _gatr_einsum_with_path = _einsum_with_path
    else:
        _gatr_einsum = torch.einsum
        _gatr_einsum_with_path = _einsum_with_path_ignored


@gatr_cache
def maximum_dtype(*args):
    """Return dtype with maximum precision."""
    dtype = max(args, key=lambda dt: torch.finfo(dt).bits)
    return dtype


@gatr_cache
def minimum_dtype(*args):
    """Return dtype with minimum precision."""
    dtype = min(args, key=lambda dt: torch.finfo(dt).bits)
    return dtype


def minimum_autocast_precision(
    min_dtype: torch.dtype = torch.float32,
    output: Optional[Union[Literal["low", "high"], torch.dtype]] = None,
    which_args: Optional[List[int]] = None,
    which_kwargs: Optional[List[str]] = None,
):
    """Decorator that ensures input tensors are autocast to a minimum precision."""

    def decorator(func: Callable):
        """Decorator that casts input tensors to minimum precision."""

        def _cast_in(var: Any):
            """Casts a single input to at least 32-bit precision."""
            if not isinstance(var, Tensor):
                return var
            if not var.dtype.is_floating_point:
                return var
            dtype = maximum_dtype(var.dtype, min_dtype)
            return var.to(dtype)

        def _cast_out(var: Any, dtype: torch.dtype):
            """Casts a single output to desired precision."""
            if not isinstance(var, Tensor):
                return var
            if not var.dtype.is_floating_point:
                return var
            return var.to(dtype)

        @wraps(func)
        def decorated_func(*args: Any, **kwargs: Any):
            """Decorated func."""
            if not (torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()):
                return func(*args, **kwargs)

            mod_args = [
                _cast_in(arg) for i, arg in enumerate(args) if which_args is None or i in which_args
            ]
            mod_kwargs = {
                key: _cast_in(val)
                for key, val in kwargs.items()
                if which_kwargs is None or key in which_kwargs
            }

            with torch.autocast(device_type="cuda", enabled=False), torch.autocast(
                device_type="cpu", enabled=False
            ):
                outputs = func(*mod_args, **mod_kwargs)

            if output is None:
                return outputs

            if output in ["low", "high"]:
                in_dtypes = [
                    arg.dtype
                    for arg in chain(args, kwargs.values())
                    if isinstance(arg, Tensor) and arg.dtype.is_floating_point
                ]
                assert len(in_dtypes)
                if output == "low":
                    out_dtype = minimum_dtype(min_dtype, *in_dtypes)
                else:
                    out_dtype = maximum_dtype(*in_dtypes)
            else:
                out_dtype = output

            if isinstance(outputs, tuple):
                return (_cast_out(val, out_dtype) for val in outputs)
            else:
                return _cast_out(outputs, out_dtype)

        return decorated_func

    return decorator


def expand_pairwise(*tensors, exclude_dims=()):
    """Expand tensors to largest, optionally excluding some axes."""
    max_dim = max(t.dim() for t in tensors)
    shapes = [(1,) * (max_dim - t.dim()) + t.shape for t in tensors]
    max_shape = [max(s[d] for s in shapes) for d in range(max_dim)]
    for d in exclude_dims:
        max_shape[d] = -1
    return tuple(t.expand(tuple(max_shape)) for t in tensors)


def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(-1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :])


def construct_reference_multivector(reference: Union[Tensor, str], inputs: Tensor) -> Tensor:
    """Constructs a reference vector for the equivariant join."""
    if reference == "data":
        # When using torch-geometric-style batching, this code should be adapted to perform the
        # mean over the items in each batch, but not over the batch dimension.
        mean_dim = tuple(range(1, len(inputs.shape) - 1))
        reference_mv = torch.mean(inputs, dim=mean_dim, keepdim=True)
    elif reference == "canonical":
        reference_mv = torch.zeros(16, device=inputs.device, dtype=inputs.dtype)
        reference_mv[..., [14, 15]] = 1.0
    else:
        if not isinstance(reference, Tensor):
            raise ValueError(
                'Reference needs to be "data", "canonical", or torch.Tensor, but found {reference}'
            )
        reference_mv = reference

    return reference_mv


def embed_scalar(scalars: torch.Tensor) -> torch.Tensor:
    """Embeds a scalar tensor into multivectors."""
    non_scalar_shape = list(scalars.shape[:-1]) + [15]
    non_scalar_components = torch.zeros(
        non_scalar_shape, device=scalars.device, dtype=scalars.dtype
    )
    embedding = torch.cat((scalars, non_scalar_components), dim=-1)

    return embedding


def embed_point(coordinates: torch.Tensor) -> torch.Tensor:
    """Embed 3D points as PGA trivectors.

    Parameters
    ----------
    coordinates : torch.Tensor with shape (..., 3)
        Cartesian point coordinates.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Point embedding in the PGA basis used by GATr.
    """
    if coordinates.shape[-1] != 3:
        raise ValueError(
            f"Point coordinates must have three components, found shape {coordinates.shape}"
        )

    batch_shape = coordinates.shape[:-1]
    multivector = torch.zeros(
        *batch_shape, 16, dtype=coordinates.dtype, device=coordinates.device
    )
    multivector[..., 14] = 1.0
    multivector[..., 13] = -coordinates[..., 0]
    multivector[..., 12] = coordinates[..., 1]
    multivector[..., 11] = -coordinates[..., 2]
    return multivector


def embed_translation(translation_vector: torch.Tensor) -> torch.Tensor:
    """Embed a 3D displacement as a PGA translator.

    The result contains the scalar component and the ideal bivector components from
    ``T(t) = 1 - 0.5 * e0 * t``.

    Parameters
    ----------
    translation_vector : torch.Tensor with shape (..., 3)
        Translation or displacement vectors.

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Translator embedding in the PGA basis used by GATr.
    """
    if translation_vector.shape[-1] != 3:
        raise ValueError(
            "Translation vectors must have three components, "
            f"found shape {translation_vector.shape}"
        )

    batch_shape = translation_vector.shape[:-1]
    multivector = torch.zeros(
        *batch_shape, 16, dtype=translation_vector.dtype, device=translation_vector.device
    )
    multivector[..., 0] = 1.0
    multivector[..., 5:8] = -0.5 * translation_vector
    return multivector


# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.




@gatr_cache
def _compute_pin_equi_linear_basis(
    device=torch.device("cpu"), dtype=torch.float32, normalize=True
) -> torch.Tensor:
    """Constructs basis elements for Pin(3,0,1)-equivariant linear maps between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype
    normalize : bool
        Whether to normalize the basis elements

    Returns
    -------
    basis : torch.Tensor with shape (7, 16, 16)
        Basis elements for equivariant linear maps.
    """

    # We constructed these manually in a notebook, here hardcoded for convenience
    basis_elements = [
        [0],
        [1, 2, 3, 4],
        [5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14],
        [15],
        [(1, 0)],
        [(5, 2), (6, 3), (7, 4)],
        [(11, 8), (12, 9), (13, 10)],
        [(15, 14)],
    ]
    basis = []

    for elements in basis_elements:
        w = torch.zeros((16, 16))
        for element in elements:
            try:
                i, j = element
                w[i, j] = 1.0
            except TypeError:
                w[element, element] = 1.0

        if normalize:
            w /= torch.linalg.norm(w)

        w = w.unsqueeze(0)
        basis.append(w)

    catted_basis = torch.cat(basis, dim=0)

    return catted_basis.to(device=device, dtype=dtype)


@gatr_cache
def _compute_reversal(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector reversal.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    reversal_diag : torch.Tensor with shape (16,)
        The diagonal of the reversal matrix, consisting of +1 and -1 entries.
    """
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@gatr_cache
def _compute_grade_involution(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector grade involution.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    involution_diag : torch.Tensor with shape (16,)
        The diagonal of the involution matrix, consisting of +1 and -1 entries.
    """
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


NUM_PIN_LINEAR_BASIS_ELEMENTS = len(_compute_pin_equi_linear_basis())


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant linear map f(x) = sum_{a,j} coeffs_a W^a_ij x_j.

    The W^a are 9 pre-defined basis elements.

    Parameters
    ----------
    x : torch.Tensor with shape (..., in_channels, 16)
        Input multivector. Batch dimensions must be broadcastable between x and coeffs.
    coeffs : torch.Tensor with shape (out_channels, in_channels, 9)
        Coefficients for the 9 basis elements. Batch dimensions must be broadcastable between x and
        coeffs.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x and coeffs.
    """
    basis = _compute_pin_equi_linear_basis(x.device, x.dtype)
    return gatr_einsum_with_path(
        "y x a, a i j, ... x j -> ... y i", coeffs, basis, x, path=[0, 1, 0, 1]
    )


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects an input tensor to the individual grades.

    The return value is a single tensor with a new grade dimension.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5, 16)
        Output multivector. The second-to-last dimension indexes the grades.
    """

    # Select kernel on correct device
    basis = _compute_pin_equi_linear_basis(x.device, x.dtype, False)

    # First five basis elements are grade projections
    basis = basis[:5]

    # Project to grades
    projections = gatr_einsum("g i j, ... j -> ... g i", basis, x)

    return projections


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Computes the reversal of a multivector.

    The reversal has the same scalar, vector, and pseudoscalar components, but flips sign in the
    bivector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """
    return _compute_reversal(x.device, x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Computes the grade involution of a multivector.

    The reversal has the same scalar, bivector, and pseudoscalar components, but flips sign in the
    vector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """

    return _compute_grade_involution(x.device, x.dtype) * x



_BILINEAR_ENTRIES = {
    "gp": (
        (0, 0, 0, 1), (0, 2, 2, 1), (0, 3, 3, 1), (0, 4, 4, 1), (0, 8, 8, -1), (0, 9, 9, -1), (0, 10, 10, -1), (0, 14, 14, -1),
        (1, 0, 1, 1), (1, 1, 0, 1), (1, 2, 5, -1), (1, 3, 6, -1), (1, 4, 7, -1), (1, 5, 2, 1), (1, 6, 3, 1), (1, 7, 4, 1),
        (1, 8, 11, -1), (1, 9, 12, -1), (1, 10, 13, -1), (1, 11, 8, -1), (1, 12, 9, -1), (1, 13, 10, -1), (1, 14, 15, 1), (1, 15, 14, -1),
        (2, 0, 2, 1), (2, 2, 0, 1), (2, 3, 8, -1), (2, 4, 9, -1), (2, 8, 3, 1), (2, 9, 4, 1), (2, 10, 14, -1), (2, 14, 10, -1),
        (3, 0, 3, 1), (3, 2, 8, 1), (3, 3, 0, 1), (3, 4, 10, -1), (3, 8, 2, -1), (3, 9, 14, 1), (3, 10, 4, 1), (3, 14, 9, 1),
        (4, 0, 4, 1), (4, 2, 9, 1), (4, 3, 10, 1), (4, 4, 0, 1), (4, 8, 14, -1), (4, 9, 2, -1), (4, 10, 3, -1), (4, 14, 8, -1),
        (5, 0, 5, 1), (5, 1, 2, 1), (5, 2, 1, -1), (5, 3, 11, 1), (5, 4, 12, 1), (5, 5, 0, 1), (5, 6, 8, -1), (5, 7, 9, -1),
        (5, 8, 6, 1), (5, 9, 7, 1), (5, 10, 15, -1), (5, 11, 3, 1), (5, 12, 4, 1), (5, 13, 14, -1), (5, 14, 13, 1), (5, 15, 10, -1),
        (6, 0, 6, 1), (6, 1, 3, 1), (6, 2, 11, -1), (6, 3, 1, -1), (6, 4, 13, 1), (6, 5, 8, 1), (6, 6, 0, 1), (6, 7, 10, -1),
        (6, 8, 5, -1), (6, 9, 15, 1), (6, 10, 7, 1), (6, 11, 2, -1), (6, 12, 14, 1), (6, 13, 4, 1), (6, 14, 12, -1), (6, 15, 9, 1),
        (7, 0, 7, 1), (7, 1, 4, 1), (7, 2, 12, -1), (7, 3, 13, -1), (7, 4, 1, -1), (7, 5, 9, 1), (7, 6, 10, 1), (7, 7, 0, 1),
        (7, 8, 15, -1), (7, 9, 5, -1), (7, 10, 6, -1), (7, 11, 14, -1), (7, 12, 2, -1), (7, 13, 3, -1), (7, 14, 11, 1), (7, 15, 8, -1),
        (8, 0, 8, 1), (8, 2, 3, 1), (8, 3, 2, -1), (8, 4, 14, 1), (8, 8, 0, 1), (8, 9, 10, -1), (8, 10, 9, 1), (8, 14, 4, 1),
        (9, 0, 9, 1), (9, 2, 4, 1), (9, 3, 14, -1), (9, 4, 2, -1), (9, 8, 10, 1), (9, 9, 0, 1), (9, 10, 8, -1), (9, 14, 3, -1),
        (10, 0, 10, 1), (10, 2, 14, 1), (10, 3, 4, 1), (10, 4, 3, -1), (10, 8, 9, -1), (10, 9, 8, 1), (10, 10, 0, 1), (10, 14, 2, 1),
        (11, 0, 11, 1), (11, 1, 8, 1), (11, 2, 6, -1), (11, 3, 5, 1), (11, 4, 15, -1), (11, 5, 3, 1), (11, 6, 2, -1), (11, 7, 14, 1),
        (11, 8, 1, 1), (11, 9, 13, -1), (11, 10, 12, 1), (11, 11, 0, 1), (11, 12, 10, -1), (11, 13, 9, 1), (11, 14, 7, -1), (11, 15, 4, 1),
        (12, 0, 12, 1), (12, 1, 9, 1), (12, 2, 7, -1), (12, 3, 15, 1), (12, 4, 5, 1), (12, 5, 4, 1), (12, 6, 14, -1), (12, 7, 2, -1),
        (12, 8, 13, 1), (12, 9, 1, 1), (12, 10, 11, -1), (12, 11, 10, 1), (12, 12, 0, 1), (12, 13, 8, -1), (12, 14, 6, 1), (12, 15, 3, -1),
        (13, 0, 13, 1), (13, 1, 10, 1), (13, 2, 15, -1), (13, 3, 7, -1), (13, 4, 6, 1), (13, 5, 14, 1), (13, 6, 4, 1), (13, 7, 3, -1),
        (13, 8, 12, -1), (13, 9, 11, 1), (13, 10, 1, 1), (13, 11, 9, -1), (13, 12, 8, 1), (13, 13, 0, 1), (13, 14, 5, -1), (13, 15, 2, 1),
        (14, 0, 14, 1), (14, 2, 10, 1), (14, 3, 9, -1), (14, 4, 8, 1), (14, 8, 4, 1), (14, 9, 3, -1), (14, 10, 2, 1), (14, 14, 0, 1),
        (15, 0, 15, 1), (15, 1, 14, 1), (15, 2, 13, -1), (15, 3, 12, 1), (15, 4, 11, -1), (15, 5, 10, 1), (15, 6, 9, -1), (15, 7, 8, 1),
        (15, 8, 7, 1), (15, 9, 6, -1), (15, 10, 5, 1), (15, 11, 4, 1), (15, 12, 3, -1), (15, 13, 2, 1), (15, 14, 1, -1), (15, 15, 0, 1),
    ),
    "outer": (
        (0, 0, 0, 1), (1, 0, 1, 1), (1, 1, 0, 1), (2, 0, 2, 1), (2, 2, 0, 1), (3, 0, 3, 1), (3, 3, 0, 1), (4, 0, 4, 1),
        (4, 4, 0, 1), (5, 0, 5, 1), (5, 1, 2, 1), (5, 2, 1, -1), (5, 5, 0, 1), (6, 0, 6, 1), (6, 1, 3, 1), (6, 3, 1, -1),
        (6, 6, 0, 1), (7, 0, 7, 1), (7, 1, 4, 1), (7, 4, 1, -1), (7, 7, 0, 1), (8, 0, 8, 1), (8, 2, 3, 1), (8, 3, 2, -1),
        (8, 8, 0, 1), (9, 0, 9, 1), (9, 2, 4, 1), (9, 4, 2, -1), (9, 9, 0, 1), (10, 0, 10, 1), (10, 3, 4, 1), (10, 4, 3, -1),
        (10, 10, 0, 1), (11, 0, 11, 1), (11, 1, 8, 1), (11, 2, 6, -1), (11, 3, 5, 1), (11, 5, 3, 1), (11, 6, 2, -1), (11, 8, 1, 1),
        (11, 11, 0, 1), (12, 0, 12, 1), (12, 1, 9, 1), (12, 2, 7, -1), (12, 4, 5, 1), (12, 5, 4, 1), (12, 7, 2, -1), (12, 9, 1, 1),
        (12, 12, 0, 1), (13, 0, 13, 1), (13, 1, 10, 1), (13, 3, 7, -1), (13, 4, 6, 1), (13, 6, 4, 1), (13, 7, 3, -1), (13, 10, 1, 1),
        (13, 13, 0, 1), (14, 0, 14, 1), (14, 2, 10, 1), (14, 3, 9, -1), (14, 4, 8, 1), (14, 8, 4, 1), (14, 9, 3, -1), (14, 10, 2, 1),
        (14, 14, 0, 1), (15, 0, 15, 1), (15, 1, 14, 1), (15, 2, 13, -1), (15, 3, 12, 1), (15, 4, 11, -1), (15, 5, 10, 1), (15, 6, 9, -1),
        (15, 7, 8, 1), (15, 8, 7, 1), (15, 9, 6, -1), (15, 10, 5, 1), (15, 11, 4, 1), (15, 12, 3, -1), (15, 13, 2, 1), (15, 14, 1, -1),
        (15, 15, 0, 1),
    ),
}


@gatr_cache
def _load_bilinear_basis(
    kind: str, device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Construct a dense algebra product basis from embedded sparse coefficients."""
    basis = torch.zeros((16, 16, 16), device=device, dtype=dtype)
    for output, left, right, value in _BILINEAR_ENTRIES[kind]:
        basis[output, left, right] = value
    return basis


def geometric_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gp = _load_bilinear_basis("gp", x.device, x.dtype)
    return gatr_einsum("i j k, ... j, ... k -> ... i", gp, x, y)


def outer_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    op = _load_bilinear_basis("outer", x.device, x.dtype)
    return gatr_einsum("i j k, ... j, ... k -> ... i", op, x, y)


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.



# Flag which reference join implementations we're using
_USE_EFFICIENT_JOIN = True


@gatr_cache
@torch.no_grad()
def _compute_dualization(
    device=torch.device("cpu"), dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constructs a tensor for the dual operation.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    permutation : list of int
        Permutation index list to compute the dual
    factors : torch.Tensor
        Signs to multiply the dual outputs with.
    """
    permutation = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    factors = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1], device=device, dtype=dtype
    )
    return permutation, factors


@gatr_cache
@torch.no_grad()
def _compute_efficient_join(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a kernel for the join operation.

    The kernel is such that join(x, y)_i = einsum(kernel_ijk, x_j, x_k).

    For now, we do this in the simplest possible way: by computing the joins between two sets of
    basis vectors. (Since the join is bilinear, that should be enough.)

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    kernel : torch.Tensor
        Joint kernel
    """

    kernel = torch.zeros((16, 16, 16), dtype=dtype, device=device)

    for i in range(16):
        for j in range(16):
            x, y = torch.zeros(16, dtype=dtype, device=device), torch.zeros(
                16, dtype=dtype, device=device
            )
            x[i] = 1.0
            y[j] = 1.0
            kernel[:, i, j] = dual(outer_product(dual(x), dual(y)))

    return kernel


@gatr_cache
@torch.no_grad()
def _compute_join_norm_idx(threshold=0.5) -> Tuple[list, list, list]:
    """Constructs everything we need to compute norm(equi_norm(x,y)) in a memory-efficient way.

    Parameters
    ----------
    threshold : float
        Threshold that determines discretization of join kernel

    Returns
    -------
    left_idx : list of list of int
        List (output components) of list (contributing terms) of indices of x
    left_idx : list of list of int
        List (output components) of list (contributing terms) of indices of y
    left_idx : list of list of float
        List (output components) of list (contributing terms) of indices of y
    """

    # Get join kernel K_ijk
    join_kernel = _compute_efficient_join()

    # Output components that contribute to the norm: all e_{i...k} without a 0 in the idx
    output_idx = [0, 2, 3, 4, 8, 9, 10, 14]

    # Identify input idx that contribute to those output idx, with the corresponding signs
    all_left_idx = []
    all_right_idx = []
    all_signs = []

    for i in output_idx:
        left_idx = []
        right_idx = []
        signs = []

        for j, k in product(range(16), repeat=2):
            if join_kernel[i, j, k] > threshold:
                left_idx.append(j)
                right_idx.append(k)
                signs.append(1.0)
            elif join_kernel[i, j, k] < -threshold:
                left_idx.append(j)
                right_idx.append(k)
                signs.append(-1.0)

        all_left_idx.append(left_idx)
        all_right_idx.append(right_idx)
        all_signs.append(signs)

    return all_left_idx, all_right_idx, all_signs


def dual(x: torch.Tensor) -> torch.Tensor:
    """Computes the dual of `inputs` (non-equivariant!).

    See Table 4 in the reference.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector, of which we want to compute the dual.

    Returns
    -------
    outputs : torch.Tensor with shale (..., 16)
        The dual of `inputs`, using the pseudoscalar component of `reference` as basis.
    """

    # Select factors on correct device
    perm, factors = _compute_dualization(x.device, x.dtype)

    # Compute dual
    result = factors * x[..., perm]

    return result


def equivariant_join(x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Computes the equivariant join.

    ```
    equivariant_join(x, y; reference) = reference_123 * dual( dual(x) ^ dual(y) )
    ```

    This function uses either explicit_equivariant_join or efficient_equivariant_join, depending
    on whether _USE_EFFICIENT_JOIN is set.

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.
    reference : torch.Tensor
        Reference multivector to break the orientation ambiguity.

    Returns
    -------
    outputs : torch.Tensor
        Equivariant join result.
    """

    if _USE_EFFICIENT_JOIN:
        return efficient_equivariant_join(x, y, reference)

    return explicit_equivariant_join(x, y, reference)


def explicit_equivariant_join(
    x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Computes the equivariant join, using the explicit, but slow, implementation.

    ```
    equivariant_join(x, y; reference) = reference_123 * dual( dual(x) ^ dual(y) )
    ```

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.
    reference : torch.Tensor
        Reference multivector to break the orientation ambiguity.

    Returns
    -------
    outputs : torch.Tensor
        Rquivariant join result.
    """
    return reference[..., [14]] * dual(outer_product(dual(x), dual(y)))


def efficient_equivariant_join(
    x: torch.Tensor, y: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Computes the equivariant join, using the efficient implementation.

    ```
    equivariant_join(x, y; reference) = reference_123 * dual( dual(x) ^ dual(y) )
    ```

    Parameters
    ----------
    x : torch.Tensor
        Left input multivector.
    y : torch.Tensor
        Right input multivector.
    reference : torch.Tensor
        Reference multivector to break the orientation ambiguity.

    Returns
    -------
    outputs : torch.Tensor
        Rquivariant join result.
    """

    kernel = _compute_efficient_join(x.device, x.dtype)
    return reference[..., [14]] * gatr_einsum("i j k , ... j, ... k -> ... i", kernel, x, y)


@minimum_autocast_precision(torch.float32)
def join_norm(
    x: torch.Tensor,
    y: torch.Tensor,
    square=False,
    channel_sum=False,
    channel_weights=None,
    epsilon=1e-6,
) -> torch.Tensor:
    """Computes the norm of the join, `|join(x,y)|`, in a single operation.

    Optionally:
    - computes the squared norm instead of the norm (when `square = True`),
    - sums over channels, meaning the second-to-last dimension (when `channel_sum = True`)

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Left input
    y : torch.Tensor with shape (..., 16)
        Right input
    square : bool
        If True, computes the squared norm rather than the norm
    channel_sum : bool
        If True, sums the result over channels (before taking the square root). We assume channels
        correspond to the second-to-last dimension of the input tensors.
    channel_weights : None torch.Tensor with shape (..., num_channels)
        If channel_sum is True, a non-None value of channel_weights weighs the different channels
        before summing over them. (Note that we do not perform any normalization here.)

    Returns
    -------
    norm : torch.Tensor with shape (..., 1)
        Norm of join
    """

    # Prepare computation
    output = 0.0
    all_left_idx, all_right_idx, all_signs = _compute_join_norm_idx()

    # Sum over all contributing terms
    for left_idx, right_idx, signs in zip(all_left_idx, all_right_idx, all_signs):
        # Compute contribution
        component = 0.0
        for j, k, sign in zip(left_idx, right_idx, signs):
            component = component + sign * x[..., j] * y[..., k]
        component = component**2

        # Compute channel sum if desired
        if channel_sum:
            if channel_weights is not None:
                component = component * channel_weights
            component = component.sum(dim=-1)

        output = output + component

    # Square root, unless the square norm is computed. The clamp avoids an infinite gradient at
    # exactly zero norm, which can otherwise make backward produce NaNs while forward stays finite.
    if not square:
        output = torch.sqrt(torch.clamp(output, min=epsilon))

    return output.unsqueeze(-1)


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.




@gatr_cache
def compute_inner_product_mask(device=torch.device("cpu")) -> torch.Tensor:
    """Constructs a bool array for the inner product calculation.

    The inner product of MVs is <~x y>_0, i.e. take the grade-0 component of the geometric
    product of the reverse of x with y.
    Both the scalar component of the GP, and the reversal matrix, are diagonal.
    Their product is 0 for basis elements involving e0, and 1 elsewhere, i.e.
    IP = [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    for dim order '', 'e0', 'e1', 'e2', 'e3', 'e01', 'e02', 'e03', 'e12', 'e13', 'e23',
                  'e012', 'e013', 'e023', 'e123', 'e0123'

    Parameters
    ----------
    device : torch.device
        Device

    Returns
    -------
    ip_mask : torch.Tensor with shape (16,)
        Inner product mask
    """
    gp = _load_bilinear_basis("gp", device, torch.float32)
    inner_product_mask = torch.diag(gp[0]) * _compute_reversal(device, torch.float32)
    return inner_product_mask.bool()


INNER_PRODUCT_INDICES = torch.arange(16)[compute_inner_product_mask()]


def inner_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the inner product of multivectors f(x,y) = <x, y> = <~x y>_0.

    Sums over the 16 multivector dimensions.

    Equal to `geometric_product(reverse(x), y)[..., [0]]` (but faster).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16) or (..., channels, 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16) or (..., channels, 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Result. Batch dimensions are result of broadcasting between x and y.
    """

    selector = INNER_PRODUCT_INDICES.to(x.device)
    x = x[..., selector]
    y = y[..., selector]

    outputs = torch.einsum("... i, ... i -> ...", x, y)

    # We want the output to have shape (..., 1)
    outputs = outputs.unsqueeze(-1)

    return outputs


@minimum_autocast_precision(torch.float32)
def norm(x: torch.Tensor) -> torch.Tensor:
    """Computes the GA norm of an input multivector.

    Equal to sqrt(inner_product(x, x)).

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Geometric algebra norm of x.
    """

    return torch.sqrt(torch.clamp(inner_product(x, x), 0.0))


def pin_invariants(x: torch.Tensor) -> torch.Tensor:
    """Computes five invariants from multivectors: scalar component, norms of the four other grades.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5)
        Invariants computed from input multivectors
    """

    # Project to grades
    projections = grade_project(x)  # (..., 5, 16)

    # Compute norms
    squared_norms = inner_product(projections, projections)[..., 0]  # (..., 5)
    norms = torch.sqrt(torch.clamp(squared_norms, 0.0))

    # Outputs: scalar component of input and norms of four other grades
    return torch.cat((x[..., [0]], norms[..., 1:]), dim=-1)  # (..., 5)


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.



@minimum_autocast_precision(torch.float32)
def equi_layer_norm(
    x: torch.Tensor, channel_dim: int = -2, gain: float = 1.0, epsilon: float = 0.01
) -> torch.Tensor:
    """Equivariant LayerNorm for multivectors.

    Rescales input such that `mean_channels |inputs|^2 = 1`, where the norm is the GA norm and the
    mean goes over the channel dimensions.

    Using a factor `gain > 1` makes up for the fact that the GP norm overestimates the actual
    standard deviation of the input data.

    Parameters
    ----------
    x : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
        Input multivectors.
    channel_dim : int
        Channel dimension index. Defaults to the second-last entry (last are the multivector
        components).
    gain : float
        Target output scale.
    epsilon : float
        Small numerical factor to avoid instabilities. By default, we use a reasonably large number
        to balance issues that arise from some multivector components not contributing to the norm.

    Returns
    -------
    outputs : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
        Normalized inputs.
    """

    # Compute mean_channels |inputs|^2
    squared_norms = inner_product(x, x)
    squared_norms = torch.mean(squared_norms, dim=channel_dim, keepdim=True)

    # Insure against low-norm tensors (which can arise even when `x.var(dim=-1)` is high b/c some
    # entries don't contribute to the inner product / GP norm!)
    squared_norms = torch.clamp(squared_norms, epsilon)

    # Rescale inputs
    outputs = gain * x / torch.sqrt(squared_norms)

    return outputs


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.


_GATED_GELU_DIV_FACTOR = math.sqrt(2 / math.pi) * 2


def gated_relu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated ReLU nonlinearity.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    ReLU(gates) * x.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes ReLU(gates) * x, with broadcasting along the last dimension.
    """

    weights = torch.nn.functional.relu(gates)
    outputs = weights * x
    return outputs


def gated_sigmoid(x: torch.Tensor, gates: torch.Tensor):
    """Pin-equivariant gated sigmoid nonlinearity.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    sigmoid(gates) * x.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes sigmoid(gates) * x, with broadcasting along the last dimension.
    """

    weights = torch.nn.functional.sigmoid(gates)
    outputs = weights * x
    return outputs


def gated_gelu(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated GeLU nonlinearity without division.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    GeLU(gates) * x.

    References
    ----------
    Dan Hendrycks, Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes GeLU(gates) * x, with broadcasting along the last dimension.
    """

    weights = torch.nn.functional.gelu(gates, approximate="tanh")
    outputs = weights * x
    return outputs


def gated_gelu_divide(x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant gated GeLU nonlinearity with division.

    Given multivector input x and scalar input gates (with matching batch dimensions), computes
    GeLU(gates) * x / gates.

    References
    ----------
    Dan Hendrycks, Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Multivector input
    gates : torch.Tensor with shape (..., 1)
        Pin-invariant gates.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Computes GeLU(gates) * x, with broadcasting along the last dimension.
    """

    weights = torch.sigmoid(_GATED_GELU_DIV_FACTOR * (gates + 0.044715 * gates**3))
    outputs = weights * x
    return outputs


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.



def grade_dropout(x: torch.Tensor, p: float, training: bool = True) -> torch.Tensor:
    """Multivector dropout, dropping out grades independently.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input data.
    p : float
        Dropout probability (assumed the same for each grade).
    training : bool
        Switches between train-time and test-time behaviour.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Inputs with dropout applied.
    """

    # Project to grades
    x = grade_project(x)

    # Apply standard 1D dropout
    # For whatever reason, that only works with a single batch dimension, so let's reshape a bit
    h = x.view(-1, 5, 16)
    h = torch.nn.functional.dropout1d(h, p=p, training=training, inplace=False)
    h = h.view(x.shape)

    # Combine grades again
    h = torch.sum(h, dim=-2)

    return h


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.



@dataclass
class SelfAttentionConfig:
    """Configuration for attention.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    num_heads : int
        Number of attention heads.
    in_s_channels : int
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int
        Output scalar channels. If None, no scalars are expected nor returned.
    additional_qk_mv_channels : int
        Whether additional multivector features for the keys and queries will be provided.
    additional_qk_s_channels : int
        Whether additional scalar features for the keys and queries will be provided.
    normalizer : str
        Normalizer function to use in sdp_dist attention
    normalizer_eps : float
        Small umerical constant for stability in the normalizer in sdp_dist attention
    multi_query: bool
        Whether to do multi-query attention
    attention_type : {"scalar", "geometric", "sdp_dist"}
        Whether the attention mechanism is based on the scalar product or also the join.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_enc_base : int
        Base for the frequencies in the positional encoding.
    output_init : str
        Initialization scheme for final linear layer
    increase_hidden_channels : int
        Factor by which to increase the number of hidden channels (both multivectors and scalars)
    dropout_prob : float or None
        Dropout probability
    """

    multi_query: bool = True
    in_mv_channels: Optional[int] = None
    out_mv_channels: Optional[int] = None
    in_s_channels: Optional[int] = None
    out_s_channels: Optional[int] = None
    num_heads: int = 8
    additional_qk_mv_channels: int = 0
    additional_qk_s_channels: int = 0
    normalizer_eps: Optional[float] = 1e-3
    pos_encoding: bool = False
    pos_enc_base: int = 4096
    output_init: str = "default"
    checkpoint: bool = True
    increase_hidden_channels: int = 2
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @property
    def hidden_mv_channels(self) -> Optional[int]:
        """Returns the number of hidden multivector channels."""

        if self.in_mv_channels is None:
            return None

        return max(self.increase_hidden_channels * self.in_mv_channels // self.num_heads, 1)

    @property
    def hidden_s_channels(self) -> Optional[int]:
        """Returns the number of hidden scalar channels."""

        if self.in_s_channels is None:
            return None

        hidden_s_channels = max(
            self.increase_hidden_channels * self.in_s_channels // self.num_heads, 4
        )

        # When using positional encoding, the number of scalar hidden channels needs to be even.
        # It also should not be too small.
        if self.pos_encoding:
            hidden_s_channels = (hidden_s_channels + 1) // 2 * 2
            hidden_s_channels = max(hidden_s_channels, 8)

        return hidden_s_channels

    @classmethod
    def cast(cls, config: Any) -> SelfAttentionConfig:
        """Casts an object as SelfAttentionConfig."""
        if isinstance(config, SelfAttentionConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.



@dataclass
class MLPConfig:
    """Geometric MLP configuration.

    Parameters
    ----------
    mv_channels : iterable of int
        Number of multivector channels at each layer, from input to output
    s_channels : None or iterable of int
        If not None, sets the number of scalar channels at each layer, from input to output. Length
        needs to match mv_channels
    activation : {"relu", "sigmoid", "gelu"}
        Which (gated) activation function to use
    dropout_prob : float or None
        Dropout probability
    """

    mv_channels: Optional[List[int]] = None
    s_channels: Optional[List[int]] = None
    activation: str = "gelu"
    dropout_prob: Optional[float] = None

    def __post_init__(self):
        """Type checking / conversion."""
        if isinstance(self.dropout_prob, str) and self.dropout_prob.lower() in ["null", "none"]:
            self.dropout_prob = None

    @classmethod
    def cast(cls, config: Any) -> MLPConfig:
        """Casts an object as MLPConfig."""
        if isinstance(config, MLPConfig):
            return config
        if isinstance(config, Mapping):
            return cls(**config)
        raise ValueError(f"Can not cast {config} to {cls}")


# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All rights reserved.
"""Pin-equivariant linear layers between multivector tensors (torch.nn.Modules)."""





class EquiLinear(nn.Module):
    """Pin-equivariant linear layer.

    The forward pass maps multivector inputs with shape (..., in_channels, 16) to multivector
    outputs with shape (..., out_channels, 16) as

    ```
    outputs[..., j, y] = sum_{i, b, x} weights[j, i, b] basis_map[b, x, y] inputs[..., i, x]
    ```

    plus an optional bias term for outputs[..., :, 0] (biases in other multivector components would
    break equivariance).

    Here basis_map are precomputed (see gatr.primitives.linear) and weights are the
    learnable weights of this layer.

    If there are auxiliary input scalars, they transform under a linear layer, and mix with the
    scalar components the multivector data. Note that in this layer (and only here) the auxiliary
    scalars are optional.

    This layer supports four initialization schemes:
     - "default":            preserves (or actually slightly reducing) the variance of the data in
                             the forward pass
     - "small":              variance of outputs is approximately one order of magnitude smaller
                             than for "default"
     - "unit_scalar":        outputs will be close to (1, 0, 0, ..., 0)
     - "almost_unit_scalar": similar to "unit_scalar", but with more stochasticity

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels
    out_mv_channels : int
        Output multivector channels
    bias : bool
        Whether a bias term is added to the scalar component of the multivector outputs
    in_s_channels : int or None
        Input scalar channels. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    initialization : {"default", "small", "unit_scalar", "almost_unit_scalar"}
        Initialization scheme. For "default", initialize with the same philosophy as most
        networks do: preserve variance (approximately) in the forward pass. For "small",
        initalize the network such that the variance of the output data is approximately one
        order of magnitude smaller than that of the input data. For "unit_scalar", initialize
        the layer such that the output multivectors will be closer to (1, 0, 0, ..., 0).
        "almost_unit_scalar" is similar, but with more randomness.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        bias: bool = True,
        initialization: str = "default",
    ) -> None:
        super().__init__()

        # Check inputs
        if initialization in ["unit_scalar", "almost_unit_scalar"]:
            assert bias, "unit_scalar and almost_unit_scalar initialization requires bias"
            if in_s_channels is None:
                raise NotImplementedError(
                    "unit_scalar and almost_unit_scalar initialization is currently only"
                    " implemented for scalar inputs"
                )

        self._in_mv_channels = in_mv_channels

        # MV -> MV
        self.weight = nn.Parameter(
            torch.empty((out_mv_channels, in_mv_channels, NUM_PIN_LINEAR_BASIS_ELEMENTS))
        )

        # We only need a separate bias here if that isn't already covered by the linear map from
        # scalar inputs
        self.bias = (
            nn.Parameter(torch.zeros((out_mv_channels, 1)))
            if bias and in_s_channels is None
            else None
        )

        # Scalars -> MV scalars
        self.s2mvs: Optional[nn.Linear]
        if in_s_channels:
            self.s2mvs = nn.Linear(in_s_channels, out_mv_channels, bias=bias)
        else:
            self.s2mvs = None

        # MV scalars -> scalars
        if out_s_channels:
            self.mvs2s = nn.Linear(in_mv_channels, out_s_channels, bias=bias)
        else:
            self.mvs2s = None

        # Scalars -> scalars
        if in_s_channels is not None and out_s_channels is not None:
            self.s2s = nn.Linear(
                in_s_channels, out_s_channels, bias=False
            )  # Bias would be duplicate
        else:
            self.s2s = None

        # Initialization
        self.reset_parameters(initialization)

        # Count nominal FLOPs
        self.nominal_flops_per_token = count_nominal_flops_in_equi_linear(
            in_mv_channels, out_mv_channels, in_s_channels, out_s_channels
        )

    def forward(
        self, multivectors: torch.Tensor, scalars: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Maps input multivectors and scalars using the most general equivariant linear map.

        The result is again multivectors and scalars.

        For multivectors we have:
        ```
        outputs[..., j, y] = sum_{i, b, x} weights[j, i, b] basis_map[b, x, y] inputs[..., i, x]
        = sum_i linear(inputs[..., i, :], weights[j, i, :])
        ```

        Here basis_map are precomputed (see gatr.primitives.linear) and weights are the
        learnable weights of this layer.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        outputs_mv = equi_linear(multivectors, self.weight)  # (..., out_channels, 16)

        if self.bias is not None:
            bias = embed_scalar(self.bias)
            outputs_mv = outputs_mv + bias

        if self.s2mvs is not None and scalars is not None:
            outputs_mv[..., 0] += self.s2mvs(scalars)

        if self.mvs2s is not None:
            outputs_s = self.mvs2s(multivectors[..., 0])
            if self.s2s is not None and scalars is not None:
                outputs_s = outputs_s + self.s2s(scalars)
        else:
            outputs_s = None

        return outputs_mv, outputs_s

    def reset_parameters(
        self,
        initialization: str,
        gain: float = 1.0,
        additional_factor=1.0 / np.sqrt(3.0),
        use_mv_heuristics=True,
    ) -> None:
        """Initializes the weights of the layer.

        Parameters
        ----------
        initialization : {"default", "small", "unit_scalar", "almost_unit_scalar"}
            Initialization scheme. For "default", initialize with the same philosophy as most
            networks do: preserve variance (approximately) in the forward pass. For "small",
            initalize the network such that the variance of the output data is approximately one
            order of magnitude smaller than that of the input data. For "unit_scalar", initialize
            the layer such that the output multivectors will be closer to (1, 0, 0, ..., 0).
            "almost_unit_scalar" is similar, but with more randomness.
        gain : float
            Gain factor for the activations. Should be 1.0 if previous layer has no activation,
            sqrt(2) if it has a ReLU activation, and so on. Can be computed with
            `torch.nn.init.calculate_gain()`.
        additional_factor : float
            Empirically, it has been found that slightly *decreasing* the data variance at each
            layer gives a better performance. In particular, the PyTorch default initialization uses
            an additional factor of 1/sqrt(3) (cancelling the factor of sqrt(3) that naturally
            arises when computing the bounds of a uniform initialization). A discussion of this was
            (to the best of our knowledge) never published, but see
            https://github.com/pytorch/pytorch/issues/57109 and
            https://soumith.ch/files/20141213_gplus_nninit_discussion.htm.
        use_mv_heuristics : bool
            Multivector components are differently affected by the equivariance constraint. If
            `use_mv_heuristics` is set to True, we initialize the weights for each output
            multivector component differently, with factors determined empirically to preserve the
            variance of each multivector component in the forward pass.
        """

        # Prefactors depending on initialization scheme
        mv_component_factors, mv_factor, mvs_bias_shift, s_factor = self._compute_init_factors(
            initialization, gain, additional_factor, use_mv_heuristics
        )

        # Following He et al, 1502.01852, we aim to preserve the variance in the forward pass.
        # A sufficient criterion for this is that the variance of the weights is given by
        # `Var[w] = gain^2 / fan`.
        # Here `gain^2` is 2 if the previous layer has a ReLU nonlinearity, 1 for the initial layer,
        # and some other value in other situations (we may not care about this too much).
        # More importantly, `fan` is the number of connections: the number of input elements that
        # get summed over to compute each output element.

        # Let us fist consider the multivector outputs.
        self._init_multivectors(mv_component_factors, mv_factor, mvs_bias_shift)

        # Then let's consider the maps to scalars.
        self._init_scalars(s_factor)

    @staticmethod
    def _compute_init_factors(initialization, gain, additional_factor, use_mv_heuristics):
        """Computes prefactors for the initialization.

        See self.reset_parameters().
        """

        if initialization not in {"default", "small", "unit_scalar", "almost_unit_scalar"}:
            raise ValueError(f"Unknown initialization scheme {initialization}")

        if initialization == "default":
            mv_factor = gain * additional_factor * np.sqrt(3)
            s_factor = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "small":
            # Change scale by a factor of 0.1 in this layer
            mv_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            s_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 0.0
        elif initialization == "unit_scalar":
            # Change scale by a factor of 0.1 for MV outputs, and initialize bias around 1
            mv_factor = 0.1 * gain * additional_factor * np.sqrt(3)
            s_factor = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 1.0
        elif initialization == "almost_unit_scalar":
            # Change scale by a factor of 0.5 for MV outputs, and initialize bias around 1
            mv_factor = 0.5 * gain * additional_factor * np.sqrt(3)
            s_factor = gain * additional_factor * np.sqrt(3)
            mvs_bias_shift = 1.0
        else:
            raise ValueError(
                f"Unknown initialization scheme {initialization}, expected"
                ' "default", "small", "unit_scalar" or "almost_unit_scalar".'
            )

        # Individual factors for each multivector component
        if use_mv_heuristics:
            # Without corrections, the variance of standard normal inputs after a forward pass
            # through this layer is different for each output grade. The reason is that the
            # equivariance constraints affect different grades differently.
            # We heuristically correct for this by initializing the weights for different basis
            # elements differently, using the following additional factors on the weight bound:
            # mv_component_factors = torch.sqrt(torch.Tensor([0.5, 4.0, 6.0, 4.0, 1.0, 0.5, 0.5]))
            mv_component_factors = torch.sqrt(
                torch.Tensor([1.0, 4.0, 6.0, 2.0, 0.5, 0.5, 1.5, 1.5, 0.5])
            )
        else:
            mv_component_factors = torch.ones(NUM_PIN_LINEAR_BASIS_ELEMENTS)
        return mv_component_factors, mv_factor, mvs_bias_shift, s_factor

    def _init_multivectors(self, mv_component_factors, mv_factor, mvs_bias_shift):
        """Weight initialization for maps to multivector outputs."""

        # We have
        # `outputs[..., j, y] = sum_{i, b, x} weights[j, i, b] basis_map[b, x, y] inputs[..., i, x]`
        # The basis maps are more or less grade projections, summing over all basis elements
        # corresponds to (almost) an identity map in the GA space. The sum over `b` and `x` thus
        # does not contribute to `fan` substantially. (We may add a small ad-hoc factor later to
        # make up for this approximation.) However, there is still the sum over incoming channels,
        # and thus `fan ~ mv_in_channels`. Assuming (for now) that the previous layer contained a
        # ReLU activation, we finally have the condition `Var[w] = 2 / mv_in_channels`.
        # Since the variance of a uniform distribution between -a and a is given by
        # `Var[Uniform(-a, a)] = a^2/3`, we should set `a = gain * sqrt(3 / mv_in_channels)`.
        # In theory (see docstring).
        fan_in = self._in_mv_channels
        bound = mv_factor / np.sqrt(fan_in)
        for i, factor in enumerate(mv_component_factors):
            nn.init.uniform_(self.weight[..., i], a=-factor * bound, b=factor * bound)

        # Now let's focus on the scalar components of the multivector outputs.
        # If there are only multivector inputs, all is good. But if scalar inputs contribute them as
        # well, they contribute to the output variance as well.
        # In this case, we initialize such that the multivector inputs and the scalar inputs each
        # contribute half to the output variance.
        # We can achieve this by inspecting the basis maps and seeing that only basis element 0
        # contributes to the scalar output. Thus, we can reduce the variance of the correponding
        # weights to give a variance of 0.5, not 1.
        if self.s2mvs is not None:
            bound = mv_component_factors[0] * mv_factor / np.sqrt(fan_in) / np.sqrt(2)
            nn.init.uniform_(self.weight[..., [0]], a=-bound, b=bound)

        # The same holds for the scalar-to-MV map, where we also just want a variance of 0.5.
        if self.s2mvs is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.s2mvs.weight
            )  # pylint:disable=protected-access
            fan_in = max(fan_in, 1)  # Since in theory we could have 0-channel scalar "data"
            bound = mv_component_factors[0] * mv_factor / np.sqrt(fan_in) / np.sqrt(2)
            nn.init.uniform_(self.s2mvs.weight, a=-bound, b=bound)

            # Bias needs to be adapted, as the overall fan in is different (need to account for MV
            # and s inputs) and we may need to account for the unit_scalar initialization scheme
            if self.s2mvs.bias is not None:
                fan_in = (
                    nn.init._calculate_fan_in_and_fan_out(self.s2mvs.weight)[0]
                    + self._in_mv_channels
                )
                bound = mv_component_factors[0] / np.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.s2mvs.bias, mvs_bias_shift - bound, mvs_bias_shift + bound)

    def _init_scalars(self, s_factor):
        """Weight initialization for maps to multivector outputs."""

        # If both exist, we need to account for overcounting again, and assign each a target a
        # variance of 0.5.
        models = []
        if self.s2s:
            models.append(self.s2s)
        if self.mvs2s:
            models.append(self.mvs2s)
        for model in models:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                model.weight
            )  # pylint:disable=protected-access
            fan_in = max(fan_in, 1)  # Since in theory we could have 0-channel scalar "data"
            bound = s_factor / np.sqrt(fan_in) / np.sqrt(len(models))
            nn.init.uniform_(model.weight, a=-bound, b=bound)
        # Bias needs to be adapted, as the overall fan in is different (need to account for MV and
        # s inputs)
        if self.mvs2s and self.mvs2s.bias is not None:
            fan_in = nn.init._calculate_fan_in_and_fan_out(self.mvs2s.weight)[
                0
            ]  # pylint:disable=protected-access
            if self.s2s:
                fan_in += nn.init._calculate_fan_in_and_fan_out(self.s2s.weight)[
                    0
                ]  # pylint:disable=protected-access
            bound = s_factor / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.mvs2s.bias, -bound, bound)


_FLOPS_PER_WEIGHT = 6
_MV_COMPONENTS = 16


def count_nominal_flops_in_equi_linear(
    in_mv_channels: int, out_mv_channels: int, in_s_channels: int, out_s_channels: int
) -> int:
    """Computes the nominal FLOPs per token for an EquiLinear layer.

    We assume:

    - the number of tokens are large, so the token-independent contraction of basis maps with
      weights has a negligible cost
    - the number of channels is large, so any biases are negligible
    - any additions are in any case negligible
    - any reshaping or transposing of the data that happens in the einsum is negligible (this is
      likely false in our implementation, but is implementation-dependent, so we don't count it)

    Then the dominant contributions come from the (weight) matrices that are multiplied with the
    scalar and multivector inputs.

    Each such matrix multiplication M_ij x_tj generates 6 FLOPs per element of M and per token in
    x.

    We verified that (in the appropriate limit) this function is in agreement with the FLOP counted
    by the deepspeed library.

    References:

    - J. Kaplan et al, "Scaling Laws for Neural Language Models", https://arxiv.org/abs/2001.08361
    - https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
    - https://www.adamcasson.com/posts/transformer-flops
    """

    in_s_channels = 0 if in_s_channels is None else in_s_channels
    out_s_channels = 0 if out_s_channels is None else out_s_channels

    s2s_flops = _FLOPS_PER_WEIGHT * in_s_channels * out_s_channels
    s2mv_flops = _FLOPS_PER_WEIGHT * in_s_channels * out_mv_channels
    mv2s_flops = _FLOPS_PER_WEIGHT * in_mv_channels * out_s_channels
    mv2mv_flops = _FLOPS_PER_WEIGHT * _MV_COMPONENTS**2 * in_mv_channels * out_mv_channels

    return s2s_flops + s2mv_flops + mv2s_flops + mv2mv_flops


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Equivariant dropout layer."""





class GradeDropout(nn.Module):
    """Grade dropout for multivectors (and regular dropout for auxiliary scalars).

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self._dropout_prob = p

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Applies dropout.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., 16)
            Multivector inputs.
        scalars : torch.Tensor
            Scalar inputs.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., 16)
            Multivector inputs with dropout applied.
        output_scalars : torch.Tensor
            Scalar inputs with dropout applied.
        """

        out_mv = grade_dropout(multivectors, p=self._dropout_prob, training=self.training)
        out_s = torch.nn.functional.dropout(scalars, p=self._dropout_prob, training=self.training)

        return out_mv, out_s


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Equivariant normalization layers."""





class EquiLayerNorm(nn.Module):
    """Equivariant LayerNorm for multivectors.

    Rescales input such that `mean_channels |inputs|^2 = 1`, where the norm is the GA norm and the
    mean goes over the channel dimensions.

    In addition, the layer performs a regular LayerNorm operation on auxiliary scalar inputs.

    Parameters
    ----------
    mv_channel_dim : int
        Channel dimension index for multivector inputs. Defaults to the second-last entry (last are
        the multivector components).
    scalar_channel_dim : int
        Channel dimension index for scalar inputs. Defaults to the last entry.
    epsilon : float
        Small numerical factor to avoid instabilities. We use a reasonably large number to balance
        issues that arise from some multivector components not contributing to the norm.
    """

    def __init__(self, mv_channel_dim=-2, scalar_channel_dim=-1, epsilon: float = 0.01):
        super().__init__()
        self.mv_channel_dim = mv_channel_dim
        self.epsilon = epsilon

        if scalar_channel_dim != -1:
            raise NotImplementedError(
                "Currently, only scalar_channel_dim = -1 is implemented, but found"
                f" {scalar_channel_dim}"
            )

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Computes equivariant LayerNorm for multivectors.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., 16)
            Multivector inputs
        scalars : torch.Tensor with shape (..., self.in_channels, self.in_scalars)
            Scalar inputs

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., 16)
            Normalized multivectors
        output_scalars : torch.Tensor with shape (..., self.out_channels, self.in_scalars)
            Normalized scalars.
        """

        outputs_mv = equi_layer_norm(
            multivectors, channel_dim=self.mv_channel_dim, epsilon=self.epsilon
        )
        normalized_shape = scalars.shape[-1:]
        outputs_s = torch.nn.functional.layer_norm(scalars, normalized_shape=normalized_shape)

        return outputs_mv, outputs_s


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Pin-equivariant geometric product layer between multivector tensors (torch.nn.Modules)."""





class GeometricBilinear(nn.Module):
    """Geometric bilinear layer.

    Pin-equivariant map between multivector tensors that constructs new geometric features via
    geometric products and the equivariant join (based on a reference vector).

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels of `x`
    out_mv_channels : int
        Output multivector channels
    hidden_mv_channels : int or None
        Hidden MV channels. If None, uses out_mv_channels.
    in_s_channels : int or None
        Input scalar channels of `x`. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: Optional[int] = None,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Default options
        if hidden_mv_channels is None:
            hidden_mv_channels = out_mv_channels

        out_mv_channels_each = hidden_mv_channels // 2
        assert (
            out_mv_channels_each * 2 == hidden_mv_channels
        ), "GeometricBilinear needs even channel number"

        # Linear projections for GP
        self.linear_left = EquiLinear(
            in_mv_channels,
            out_mv_channels_each,
            in_s_channels=in_s_channels,
            out_s_channels=None,
        )
        self.linear_right = EquiLinear(
            in_mv_channels,
            out_mv_channels_each,
            in_s_channels=in_s_channels,
            out_s_channels=None,
            initialization="almost_unit_scalar",
        )

        # Linear projections for join
        self.linear_join_left = EquiLinear(
            in_mv_channels, out_mv_channels_each, in_s_channels=in_s_channels, out_s_channels=None
        )
        self.linear_join_right = EquiLinear(
            in_mv_channels, out_mv_channels_each, in_s_channels=in_s_channels, out_s_channels=None
        )

        # Output linear projection
        self.linear_out = EquiLinear(
            hidden_mv_channels, out_mv_channels, in_s_channels, out_s_channels
        )

    def forward(
        self,
        multivectors: torch.Tensor,
        reference_mv: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors
        scalars : torch.Tensor with shape (..., in_s_channels)
            Input scalars
        reference_mv : torch.Tensor with shape (..., 16)
            Reference multivector for equivariant join.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., self.out_mv_channels, 16)
            Output multivectors
        output_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars.
        """

        # GP
        left, _ = self.linear_left(multivectors, scalars=scalars)
        right, _ = self.linear_right(multivectors, scalars=scalars)
        gp_outputs = geometric_product(left, right)

        # Equivariant join
        left, _ = self.linear_join_left(multivectors, scalars=scalars)
        right, _ = self.linear_join_right(multivectors, scalars=scalars)
        join_outputs = equivariant_join(left, right, reference_mv)

        # Output linear
        outputs_mv = torch.cat((gp_outputs, join_outputs), dim=-2)
        outputs_mv, outputs_s = self.linear_out(outputs_mv, scalars=scalars)

        return outputs_mv, outputs_s


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.




class ScalarGatedNonlinearity(nn.Module):
    """Gated nonlinearity, where the gate is simply given by the scalar component of the input.

    Given multivector input x, computes f(x_0) * x, where f can either be ReLU, sigmoid, or GeLU.

    Auxiliary scalar inputs are simply processed with ReLU, sigmoid, or GeLU, without gating.

    Parameters
    ----------
    nonlinearity : {"relu", "sigmoid", "gelu"}
        Non-linearity type
    """

    def __init__(self, nonlinearity: str = "relu", **kwargs) -> None:
        super().__init__()

        gated_fn_dict = dict(relu=gated_relu, gelu=gated_gelu, sigmoid=gated_sigmoid)
        scalar_fn_dict = dict(
            relu=nn.functional.relu, gelu=nn.functional.gelu, sigmoid=nn.functional.sigmoid
        )
        try:
            self.gated_nonlinearity = gated_fn_dict[nonlinearity]
            self.scalar_nonlinearity = scalar_fn_dict[nonlinearity]
        except KeyError as exc:
            raise ValueError(
                f"Unknown nonlinearity {nonlinearity} for options {list(gated_fn_dict.keys())}"
            ) from exc

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes f(x_0) * x for multivector x, where f is GELU, ReLU, or sigmoid.

        f is chosen depending on self.nonlinearity.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., self.in_channels, 16)
            Input multivectors
        scalars : None or torch.Tensor with shape (..., self.in_channels, self.in_scalars)
            Input scalars

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., self.out_channels, 16)
            Output multivectors
        output_scalars : torch.Tensor with shape (..., self.out_channels, self.in_scalars)
            Output scalars
        """

        gates = multivectors[..., [0]]
        outputs_mv = self.gated_nonlinearity(multivectors, gates=gates)
        outputs_s = self.scalar_nonlinearity(scalars)

        return outputs_mv, outputs_s


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Factory functions for simple MLPs for multivector data."""





class GeoMLP(nn.Module):
    """Geometric MLP.

    This is a core component of GATr's transformer blocks. It is similar to a regular MLP, except
    that it uses geometric bilinears (GP and equivariant join) in place of the first linear layer.

    Assumes input has shape `(..., channels[0], 16)`, output has shape `(..., channels[-1], 16)`,
    will create hidden layers with shape `(..., channel, 16)` for each additional entry in
    `channels`.

    Parameters
    ----------
    config: MLPConfig
        Configuration object
    """

    def __init__(
        self,
        config: MLPConfig,
    ) -> None:
        super().__init__()

        # Store settings
        self.config = config

        assert config.mv_channels is not None
        s_channels = (
            [None for _ in config.mv_channels] if config.s_channels is None else config.s_channels
        )

        layers: List[nn.Module] = []

        if len(config.mv_channels) >= 2:
            layers.append(
                GeometricBilinear(
                    in_mv_channels=config.mv_channels[0],
                    out_mv_channels=config.mv_channels[1],
                    in_s_channels=s_channels[0],
                    out_s_channels=s_channels[1],
                )
            )
            if config.dropout_prob is not None:
                layers.append(GradeDropout(config.dropout_prob))

            for in_, out, in_s, out_s in zip(
                config.mv_channels[1:-1], config.mv_channels[2:], s_channels[1:-1], s_channels[2:]
            ):
                layers.append(ScalarGatedNonlinearity(config.activation))
                layers.append(EquiLinear(in_, out, in_s_channels=in_s, out_s_channels=out_s))
                if config.dropout_prob is not None:
                    layers.append(GradeDropout(config.dropout_prob))

        self.layers = nn.ModuleList(layers)

    def forward(
        self, multivectors: torch.Tensor, scalars: torch.Tensor, reference_mv: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        reference_mv : torch.Tensor with shape (..., 16)
            Reference multivector for equivariant join.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        mv, s = multivectors, scalars

        for i, layer in enumerate(self.layers):
            if i == 0:
                mv, s = layer(mv, scalars=s, reference_mv=reference_mv)
            else:
                mv, s = layer(mv, scalars=s)

        return mv, s


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

"""Adapted from the below.

https://github.com/EleutherAI/gpt-neox/blob/737c9134bfaff7b58217d61f6619f1dcca6c484f/megatron/model/positional_embeddings.py
by EleutherAI at https://github.com/EleutherAI/gpt-neox

Copyright (c) 2021, EleutherAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



class ApplyRotaryPositionalEncoding(torch.nn.Module):
    """Applies rotary position encodings (RoPE) to scalar tensors.

    References
    ----------
    Jianlin Su et al, "RoFormer: Enhanced Transformer with Rotary Position Embedding",
        arXiv:2104.09864

    Parameters
    ----------
    num_channels : int
        Number of channels (key and query size).
    item_dim : int
        Embedding dimension. Should be even.
    base : int
        Determines the frequencies.
    """

    def __init__(self, num_channels, item_dim, base=4096):
        super().__init__()

        assert (
            num_channels % 2 == 0
        ), "Number of channels needs to be even for rotary position embeddings"

        inv_freq = 1.0 / (base ** (torch.arange(0, num_channels, 2).float() / num_channels))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.device_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.item_dim = item_dim
        self.num_channels = num_channels

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """Computes rotary embeddings along `self.item_dim` and applies them to inputs.

        The inputs are usually scalar queries and keys.

        Assumes that the last dimension is the feature dimension (and is thus not suited
        for multivector data!).

        Parameters
        ----------
        scalars : torch.Tensor of shape (..., num_channels)
            Input data. The last dimension is assumed to be the channel / feature dimension
            (NOT the 16 dimensions of a multivector).

        Returns
        -------
        outputs : torch.Tensor of shape (..., num_channels)
            Output data. Rotary positional embeddings applied to the input tensor.
        """

        # Check inputs
        assert scalars.shape[-1] == self.num_channels

        # Compute embeddings, if not already cached
        self._compute_embeddings(scalars)

        # Apply embeddings
        outputs = scalars * self.cos_cached + self._rotate_half(scalars) * self.sin_cached

        return outputs

    def _compute_embeddings(self, inputs):
        """Computes position embeddings and stores them.

        The position embedding is computed along dimension `item_dim` of tensor `inputs`
        and is stored in `self.sin_cached` and `self.cos_cached`.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data.
        """
        seq_len = inputs.shape[self.item_dim]
        if seq_len != self.seq_len_cached or inputs.device != self.device_cached:
            self.seq_len_cached = seq_len
            self.device_cached = inputs.device
            t = torch.arange(inputs.shape[self.item_dim], device=inputs.device).type_as(
                self.inv_freq
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(inputs.device)

            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

            # Insert appropriate amount of dimensions such that the embedding correctly enumerates
            # along the item dim
            item_dim = (
                self.item_dim if self.item_dim >= 0 else inputs.ndim + self.item_dim
            )  # Deal with item_dim < 0
            for _ in range(item_dim + 1, inputs.ndim - 1):
                self.cos_cached = self.cos_cached.unsqueeze(1)
                self.sin_cached = self.sin_cached.unsqueeze(1)

    @staticmethod
    def _rotate_half(inputs):
        """Utility function that "rotates" a tensor, as required for rotary embeddings."""
        x1, x2 = inputs[..., : inputs.shape[-1] // 2], inputs[..., inputs.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.



class QKVModule(nn.Module):
    """Compute (multivector and scalar) queries, keys, and values via multi-head attention.

    Parameters
    ----------
    config: SelfAttentionConfig
        Attention configuration
    """

    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self.in_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=3 * config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=None
            if config.in_s_channels is None
            else 3 * config.hidden_s_channels * config.num_heads,
        )
        self.config = config

    def forward(
        self, inputs, scalars, additional_qk_features_mv=None, additional_qk_features_s=None
    ):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Multivector inputs
        scalars : torch.Tensor
            Scalar inputs
        additional_qk_features_mv : None or torch.Tensor
            Additional multivector features that should be provided for the Q/K computation (e.g.
            positions of objects)
        additional_qk_features_s : None or torch.Tensor
            Additional scalar features that should be provided for the Q/K computation (e.g.
            object types)

        Returns
        -------
        q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
            Queries, multivector part.
        k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
            Keys, multivector part.
        v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
            Values, multivector part.
        q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
            Queries, scalar part.
        k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
            Keys, scalar part.
        v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
            Values, scalar part.
        """

        # Additional inputs
        if additional_qk_features_mv is not None:
            inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        if additional_qk_features_s is not None:
            scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)

        qkv_mv, qkv_s = self.in_linear(
            inputs, scalars
        )  # (..., num_items, 3 * hidden_channels * num_heads, 16)
        qkv_mv = rearrange(
            qkv_mv,
            "... items (qkv hidden num_heads) x -> qkv ... num_heads items hidden x",
            num_heads=self.config.num_heads,
            hidden=self.config.hidden_mv_channels,
            qkv=3,
        )
        q_mv, k_mv, v_mv = qkv_mv  # each: (..., num_heads, num_items, num_channels, 16)

        # Same, for optional scalar components
        if qkv_s is not None:
            qkv_s = rearrange(
                qkv_s,
                "... items (qkv hidden num_heads) -> qkv ... num_heads items hidden",
                num_heads=self.config.num_heads,
                hidden=self.config.hidden_s_channels,
                qkv=3,
            )
            q_s, k_s, v_s = qkv_s  # each: (..., num_heads, num_items, num_channels)
        else:
            q_s, k_s, v_s = None, None, None

        return q_mv, k_mv, v_mv, q_s, k_s, v_s


class MultiQueryQKVModule(nn.Module):
    """Compute (multivector and scalar) queries, keys, and values via multi-query attention.

    Parameters
    ----------
    config: SelfAttentionConfig
        Attention configuration
    """

    def __init__(self, config: SelfAttentionConfig):
        super().__init__()

        # Q projection
        self.q_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels * config.num_heads,
        )

        # Key and value projections (shared between heads)
        self.k_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels + config.additional_qk_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels + config.additional_qk_s_channels,
            out_s_channels=config.hidden_s_channels,
        )
        self.v_linear = EquiLinear(
            in_mv_channels=config.in_mv_channels,
            out_mv_channels=config.hidden_mv_channels,
            in_s_channels=config.in_s_channels,
            out_s_channels=config.hidden_s_channels,
        )
        self.config = config

    def forward(
        self, inputs, scalars, additional_qk_features_mv=None, additional_qk_features_s=None
    ):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Multivector inputs
        scalars : torch.Tensor
            Scalar inputs
        additional_qk_features_mv : None or torch.Tensor
            Additional multivector features that should be provided for the Q/K computation (e.g.
            positions of objects)
        additional_qk_features_s : None or torch.Tensor
            Additional scalar features that should be provided for the Q/K computation (e.g.
            object types)

        Returns
        -------
        q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
            Queries, multivector part.
        k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
            Keys, multivector part.
        v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
            Values, multivector part.
        q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
            Queries, scalar part.
        k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
            Keys, scalar part.
        v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
            Values, scalar part.
        """

        # Additional inputs
        if additional_qk_features_mv is not None:
            qk_inputs = torch.cat((inputs, additional_qk_features_mv), dim=-2)
        else:
            qk_inputs = inputs
        if scalars is not None and additional_qk_features_s is not None:
            qk_scalars = torch.cat((scalars, additional_qk_features_s), dim=-1)
        else:
            qk_scalars = scalars

        # Project to queries, keys, and values (multivector reps)
        q_mv, q_s = self.q_linear(
            qk_inputs, qk_scalars
        )  # (..., num_items, hidden_channels * num_heads, 16)
        k_mv, k_s = self.k_linear(qk_inputs, qk_scalars)  # (..., num_items, hidden_channels, 16)
        v_mv, v_s = self.v_linear(inputs, scalars)  # (..., num_items, hidden_channels, 16)

        # Rearrange to (..., heads, items, channels, 16) shape
        q_mv = rearrange(
            q_mv,
            "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
            num_heads=self.config.num_heads,
            hidden_channels=self.config.hidden_mv_channels,
        )
        k_mv = rearrange(k_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")
        v_mv = rearrange(v_mv, "... items hidden_channels x -> ... 1 items hidden_channels x")

        # Same for scalars
        if q_s is not None:
            q_s = rearrange(
                q_s,
                "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_s_channels,
            )
            k_s = rearrange(k_s, "... items hidden_channels -> ... 1 items hidden_channels")
            v_s = rearrange(v_s, "... items hidden_channels -> ... 1 items hidden_channels")
        else:
            q_s, k_s, v_s = None, None, None

        return q_mv, k_mv, v_mv, q_s, k_s, v_s


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.



# When computing the normalization factor in attention weights, multivectors contribute with the
# following factor:
_MV_SIZE_FACTOR = 8

# Multivector indices that contribute to inner product and trivectors
# All components that contribute to the inner product:
_INNER_PRODUCT_IDX = [0, 2, 3, 4, 8, 9, 10, 14]
# Scalar, non-ideal part of vector and bivector; no trivectors:
_INNER_PRODUCT_WO_TRI_IDX = [0, 2, 3, 4, 8, 9, 10]
# Trivector indices (ideal part first):
_TRIVECTOR_IDX = [11, 12, 13, 14]

# Masked out attention logits are set to this constant (a finite replacement for -inf):
_MASKED_OUT = float("-inf")

# Force the use of xformers attention, even when no xformers attention mask is provided:
FORCE_XFORMERS = True


def sdp_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., num_items_in, num_s_channels_out)
        Values, scalar part.

    Returns
    -------
    outputs_mv : Tensor with shape (..., num_items_out, num_mv_channels_out, 16)
        Result, multivector part
    outputs_s : Tensor with shape (..., num_items_out, num_s_channels_out)
        Result, scalar part
    """

    # Construct queries and keys by concatenating relevant MV components and aux scalars
    q = torch.cat([rearrange(q_mv[..., _INNER_PRODUCT_IDX], "... c x -> ... (c x)"), q_s], -1)
    k = torch.cat([rearrange(k_mv[..., _INNER_PRODUCT_IDX], "... c x -> ... (c x)"), k_s], -1)

    num_channels_out = v_mv.shape[-2]
    v = torch.cat([rearrange(v_mv, "... c x -> ... (c x)"), v_s], -1)

    v_out = scaled_dot_product_attention(q, k, v)

    v_out_mv = rearrange(v_out[..., : num_channels_out * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_channels_out * 16 :]

    return v_out_mv, v_out_s


def pga_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
    weights: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    attention_mask=None,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products and the equivariant join.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + norm(join(q_mv[..., i, :, :], k_mv[..., j, :, :]))
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Optionally, the three contributions are weighted with `weights`.

    This is not used in GATr, because it does not reduce to dot-product attention and thus does not
    benefit from efficient implementations like `geometric_attention()` does.

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., num_items_in, num_s_channels_out)
        Values, scalar part.
    weights : None, or tuple of three Tensors
        Weights for the combination of the inner product, join, and aux scalar parts
    attention_mask: None or Tensor with shape (..., num_items, num_items)
        Optional attention mask

    Returns
    -------
    outputs_mv : Tensor with shape (..., num_items_out, num_mv_channels_out, 16)
        Result, multivector part.
    outputs_s : Tensor with shape (..., num_items_out, num_s_channels_out)
        Result, scalar part.
    """

    # Negative weights are trouble
    if weights is not None:
        for weight in weights:
            assert torch.min(weight) >= 0.0

    # Compute attention weights, first through the inner product between multivectors...
    q_mv = rearrange(q_mv, "... items_out channels x -> ... items_out 1 channels x")
    k_mv = rearrange(k_mv, "... items_in channels x -> ... 1 items_in channels x")
    h = inner_product(q_mv, k_mv)[..., 0]  # (..., items_out, items_in, channels)
    if weights is not None:
        h = weights[0] * h
    attn_weights = torch.sum(h, dim=-1)  # (..., items_out, items_in)

    # ... then through the join...
    h = -join_norm(
        q_mv, k_mv, channel_sum=True, channel_weights=weights[1] if weights is not None else None
    )[
        ..., 0
    ]  # (..., items_out, items_in)
    attn_weights = attn_weights + h

    # ... and finally from auxiliary scalars
    q_s = rearrange(q_s, "... items_out channels -> ... items_out 1 channels")
    k_s = rearrange(k_s, "... items_in channels -> ... 1 items_in channels")
    h = q_s * k_s  # (..., items_out, items_in, channels)
    if weights is not None:
        h = weights[2] * h
    attn_weights = attn_weights + torch.sum(h, dim=-1)  # (..., items_out, items_in)

    # Attention mask
    if attention_mask is not None:
        attn_weights.masked_fill_(~attention_mask, _MASKED_OUT)

    # Combine and weight
    attn_weights = attn_weights / np.sqrt(2 * q_mv.shape[-2] * _MV_SIZE_FACTOR + q_s.shape[-1])

    # Softmax
    attn_weights = attn_weights.softmax(dim=-1)  # Softmax over items_in

    # Compute attention output
    outputs_mv = torch.einsum(
        "... j i, ... i c x -> ... j c x", attn_weights, v_mv
    )  # (..., items_out, channels, 16)
    outputs_s = torch.einsum(
        "... j i, ... i c -> ... j c", attn_weights, v_s
    )  # (..., items_out, channels)

    return outputs_mv, outputs_s


@gatr_cache
def _build_dist_basis(device, dtype) -> Tuple[Tensor, Tensor]:
    """Compute basis features for queries and keys in the geometric SDP attention.

    Parameters
    ----------
    device: torch.device
        Device.
    dtype: torch.dtype
        Dtype.

    Returns
    -------
    basis_q : Tensor with shape (4, 4, 5)
        Basis features for queries.
    basis_k : Tensor with shape (4, 4, 5)
        Basis features for keys.
    """
    r3 = torch.arange(3, device=device)
    basis_q = torch.zeros((4, 4, 5), device=device, dtype=dtype)
    basis_k = torch.zeros((4, 4, 5), device=device, dtype=dtype)

    # -sum_i (q_i^2) * k_0^2
    basis_q[r3, r3, 0] = 1
    basis_k[3, 3, 0] = -1

    # -q_0^2 * sum_i (k_i^2)
    basis_q[3, 3, 1] = 1
    basis_k[r3, r3, 1] = -1

    # sum_i 2 q_0 q_i k_0 k_i
    basis_q[r3, 3, 2 + r3] = 1
    basis_k[r3, 3, 2 + r3] = 2

    return basis_q, basis_k


def _build_dist_vec(tri: Tensor, basis: Tensor, normalizer: Callable[[Tensor], Tensor]) -> Tensor:
    """Build 5D vector whose inner product with another such vector computes the squared distance.

    Parameters
    ----------
    tri: Tensor
        Batch of multivectors, only trivector part is used.
    basis: Tensor
        One of the bases from _build_dist_basis.
    normalizer: Callable[[Tensor], Tensor]
        A normalization function.

    Returns
    -------
    outputs : Tensor
        Batch of 5D vectors
    """
    tri_normed = tri * normalizer(tri[..., [3]])
    vec = gatr_einsum("xyz,...x,...y->...z", basis, tri_normed, tri_normed)
    return vec


@minimum_autocast_precision(torch.float32, output="low")
def _lin_square_normalizer(v: Tensor, epsilon=0.001) -> Tensor:
    """Apply linear square normalization to the input tensor.

    Parameters
    ----------
    v : Tensor
        Input tensor.
    epsilon : float, optional
        Small constant added to the denominator to avoid division by zero.
        Default is 0.001.

    Returns
    -------
    normalized_v : Tensor
        Normalized tensor after applying linear square normalization.
    """
    return v / (v.pow(2) + epsilon)


def geometric_attention(
    q_mv: Tensor,
    k_mv: Tensor,
    v_mv: Tensor,
    q_s: Tensor,
    k_s: Tensor,
    v_s: Tensor,
    normalizer: Callable[[Tensor], Tensor],
    weights: Optional[Tensor] = None,
    attn_mask: Optional[Union[AttentionBias, Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Equivariant geometric attention based on scaled dot products and nonlinear aux features.

    This is the main attention mechanism used in GATr. Thanks to the nonlinear features, the
    scaled-dot-product attention takes into account the Euclidean distance.

    Expects both multivector and scalar queries, keys, and values as inputs.
    Then this function computes multivector and scalar outputs in the following way:

    ```
    attn_weights[..., i, j] = softmax_j[
        pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
        + inner_product(phi(q_s[..., i, :]), psi(k_s[..., j, :]))
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Optionally, the three contributions are weighted with `weights`.

    Parameters
    ----------
    q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
        Queries, multivector part.
    k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
        Keys, multivector part.
    v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
        Values, multivector part.
    q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
        Queries, scalar part.
    k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
        Keys, scalar part.
    v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
        Values, scalar part.
    normalizer : callable
        Normalization function.
    weights: Optional[Tensor] with shape (..., 1, num_channels_in)
        Weights for the combination of the inner product, nonlinear distance-aware features, and
        scalar parts.
    attn_mask: None or AttentionBias or Tensor with shape (..., num_items_in, num_items_out)
        Optional attention mask. If provided as a tensor, it should be of either of shape
        `(num_items_in, num_items_out)`, `(..., 1, num_items_in, num_items_out)`, or
        `(..., num_heads, num_items_in, num_items_out)`.

    Returns
    -------
    outputs_mv : Tensor with shape (..., heads, num_items_out, num_channels_out, 16)
        Result, multivector part.
    outputs_s : Tensor with shape (..., heads, num_items_out, num_s_channels_out)
        Result, scalar part.
    """
    bh_shape = q_mv.shape[:-3]
    q_mv = to_nd(q_mv, 5)
    k_mv = to_nd(k_mv, 5)
    v_mv = to_nd(v_mv, 5)
    q_s = to_nd(q_s, 4)
    k_s = to_nd(k_s, 4)
    v_s = to_nd(v_s, 4)

    if isinstance(attn_mask, Tensor) and len(attn_mask.shape) > 2:
        # Attention mask tensors should be reshaped to [-1, heads or 1, q_tokens, k_tokens]
        attn_mask = attn_mask.view(-1, *attn_mask.shape[-3:])

    num_mv_channels_v = v_mv.shape[-2]
    num_s_channels_v = v_s.shape[-1]
    num_mv_channels_qk = q_mv.shape[-2]
    num_s_channels_qk = q_s.shape[-1]

    q_tri = q_mv[..., _TRIVECTOR_IDX]
    k_tri = k_mv[..., _TRIVECTOR_IDX]

    basis_q, basis_k = _build_dist_basis(q_tri.device, q_tri.dtype)

    q_dist = _build_dist_vec(q_tri, basis_q, normalizer)
    k_dist = _build_dist_vec(k_tri, basis_k, normalizer)
    if weights is not None:
        q_dist = q_dist * weights[..., None].to(q_dist.dtype)

    device = q_mv.device
    dtype = q_mv.dtype

    num_channels_qk = num_mv_channels_qk * (7 + 5) + num_s_channels_qk
    num_channels_v = num_mv_channels_v * 16 + num_s_channels_v
    num_channels = max(num_channels_qk, num_channels_v)
    num_channels = 8 * -(-num_channels // 8)  # Ceil to multiple of 8

    q = torch.cat(
        [
            rearrange(q_mv[..., _INNER_PRODUCT_WO_TRI_IDX], "... c x -> ... (c x)"),
            rearrange(q_dist, "... c d -> ... (c d)"),
            q_s,
            torch.zeros(*q_s.shape[:3], num_channels - num_channels_qk, device=device, dtype=dtype),
        ],
        -1,
    )
    k = torch.cat(
        [
            rearrange(k_mv[..., _INNER_PRODUCT_WO_TRI_IDX], "... c x -> ... (c x)"),
            rearrange(k_dist, "... c d -> ... (c d)"),
            k_s,
            torch.zeros(*k_s.shape[:3], num_channels - num_channels_qk, device=device, dtype=dtype),
        ],
        -1,
    )

    v = torch.cat(
        [
            rearrange(v_mv, "... c x -> ... (c x)"),
            v_s,
            torch.zeros(*v_s.shape[:3], num_channels - num_channels_v, device=device, dtype=dtype),
        ],
        -1,
    )
    k = k * math.sqrt(num_channels / num_channels_qk)  # Correct for zero padding
    q, k, v_out = _sdpa_graph_breaking(q, k, v, attn_mask=attn_mask)

    v_out_mv = rearrange(v_out[..., : num_mv_channels_v * 16], "... (c x) -> ...  c x", x=16)
    v_out_s = v_out[..., num_mv_channels_v * 16 : num_mv_channels_v * 16 + num_s_channels_v]

    v_out_mv = v_out_mv.view(*bh_shape, *v_out_mv.shape[-3:])
    v_out_s = v_out_s.view(*bh_shape, *v_out_s.shape[-2:])

    return v_out_mv, v_out_s


@torch.compiler.disable
def _sdpa_graph_breaking(q, k, v, attn_mask):
    """A helper function to isolate the graph-breaking parts of the attention (cf. decorator).

    TODO: This function can be dissolved once we get expand_pairwise to not break the graph;
    then we can simply compiler.disable the xformers attention.

    """
    q, k, v = expand_pairwise(q, k, v, exclude_dims=(-2,))  # Don't expand along token dimension)
    v_out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return q, k, v_out


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Union[AttentionBias, Tensor]] = None,
) -> Tensor:
    """Execute (vanilla) scaled dot-product attention.

    Dynamically dispatch to xFormers if attn_mask is an instance of xformers.ops.AttentionBias
    or FORCE_XFORMERS is set, use torch otherwise.

    Parameters
    ----------
    query : Tensor
        of shape [batch, head, item, d]
    key : Tensor
        of shape [batch, head, item, d]
    value : Tensor
        of shape [batch, head, item, d]
    attn_mask : Optional[Union[AttentionBias, Tensor]]
        Attention mask

    Returns
    -------
    Tensor
        of shape [batch, head, item, d]
    """
    if FORCE_XFORMERS or isinstance(attn_mask, AttentionBias):
        # [batch, head, item, d] -> [batch, item, head, d]
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        out = memory_efficient_attention(
            query.contiguous(), key.contiguous(), value, attn_bias=attn_mask
        )
        out = out.transpose(1, 2)  # [batch, item, head, d] -> [batch, head, item, d]
        return out
    return torch_sdpa(query, key, value, attn_mask=attn_mask)


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Self-attention layers."""





class GeometricAttention(nn.Module):
    """Geometric attention layer.

    This is the main attention mechanism used in GATr. Thanks to the nonlinear features, the
    scaled-dot-product attention takes into account the Euclidean distance.

    Given multivector and scalar queries, keys, and values, this layer computes:

    ```
    attn_weights[..., i, j] = softmax_j[
        weights[0] * pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
        + weights[1] * inner_product(phi(q_s[..., i, :]), psi(k_s[..., j, :]))
        + weights[2] * euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
    ]
    out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
    out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
    ```

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

        self.normalizer = partial(_lin_square_normalizer, epsilon=config.normalizer_eps)
        self.log_weights = nn.Parameter(
            torch.zeros((config.num_heads, 1, config.hidden_mv_channels))
        )

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        """Forward pass through geometric attention.

        Given multivector and scalar queries, keys, and values, this forward pass computes:

        ```
        attn_weights[..., i, j] = softmax_j[
            weights[0] * pga_inner_product(q_mv[..., i, :, :], k_mv[..., j, :, :])
            + weights[1] * inner_product(phi(q_s[..., i, :]), psi(k_s[..., j, :]))
            + weights[2] * euclidean_inner_product(q_s[..., i, :], k_s[..., j, :])
        ]
        out_mv[..., i, c, :] = sum_j attn_weights[..., i, j] v_mv[..., j, c, :] / norm
        out_s[..., i, c] = sum_j attn_weights[..., i, j] v_s[..., j, c] / norm
        ```

        Parameters
        ----------
        q_mv : Tensor with shape (..., num_items_out, num_mv_channels_in, 16)
            Queries, multivector part.
        k_mv : Tensor with shape (..., num_items_in, num_mv_channels_in, 16)
            Keys, multivector part.
        v_mv : Tensor with shape (..., num_items_in, num_mv_channels_out, 16)
            Values, multivector part.
        q_s : Tensor with shape (..., heads, num_items_out, num_s_channels_in)
            Queries, scalar part.
        k_s : Tensor with shape (..., heads, num_items_in, num_s_channels_in)
            Keys, scalar part.
        v_s : Tensor with shape (..., heads, num_items_in, num_s_channels_out)
            Values, scalar part.
        attention_mask: None or Tensor or AttentionBias
            Optional attention mask.
        """

        weights = self.log_weights.exp()
        h_mv, h_s = geometric_attention(
            q_mv,
            k_mv,
            v_mv,
            q_s,
            k_s,
            v_s,
            normalizer=self.normalizer,
            weights=weights,
            attn_mask=attention_mask,
        )

        return h_mv, h_s


# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Self-attention layers."""





class SelfAttention(nn.Module):
    """Geometric self-attention layer.

    Constructs queries, keys, and values, computes attention, and projects linearly to outputs.

    Parameters
    ----------
    config : SelfAttentionConfig
        Attention configuration.
    """

    def __init__(self, config: SelfAttentionConfig) -> None:
        super().__init__()

        # Store settings
        self.config = config

        # QKV computation
        self.qkv_module = MultiQueryQKVModule(config) if config.multi_query else QKVModule(config)

        # Output projection
        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=(
                None
                if config.in_s_channels is None
                else config.hidden_s_channels * config.num_heads
            ),
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        # Optional positional encoding
        self.pos_encoding: nn.Module
        if config.pos_encoding:
            self.pos_encoding = ApplyRotaryPositionalEncoding(
                config.hidden_s_channels, item_dim=-2, base=config.pos_enc_base
            )
        else:
            self.pos_encoding = nn.Identity()

        # Attention
        self.attention = GeometricAttention(config)

        # Dropout
        self.dropout: Optional[nn.Module]
        if config.dropout_prob is not None:
            self.dropout = GradeDropout(config.dropout_prob)
        else:
            self.dropout = None

    def forward(
        self,
        multivectors: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward pass on inputs with shape `(..., items, channels, 16)`.

        The result is the following:

        ```
        # For each head
        queries = linear_channels(inputs)
        keys = linear_channels(inputs)
        values = linear_channels(inputs)
        hidden = attention_items(queries, keys, values, biases=biases)
        head_output = linear_channels(hidden)

        # Combine results
        output = concatenate_heads head_output
        ```

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., num_items, channels_in, 16)
            Input multivectors.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        scalars : None or torch.Tensor with shape (..., num_items, num_items, in_scalars)
            Optional input scalars
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., num_items, channels_out, 16)
            Output multivectors.
        output_scalars : torch.Tensor with shape (..., num_items, channels_out, out_scalars)
            Output scalars, if scalars are provided. Otherwise None.
        """
        # Compute Q, K, V
        q_mv, k_mv, v_mv, q_s, k_s, v_s = self.qkv_module(
            multivectors, scalars, additional_qk_features_mv, additional_qk_features_s
        )

        # Rotary positional encoding
        q_s = self.pos_encoding(q_s)
        k_s = self.pos_encoding(k_s)

        # Attention layer
        h_mv, h_s = self.attention(q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attention_mask)

        h_mv = rearrange(
            h_mv, "... n_heads n_items hidden_channels x -> ... n_items (n_heads hidden_channels) x"
        )
        h_s = rearrange(
            h_s, "... n_heads n_items hidden_channels -> ... n_items (n_heads hidden_channels)"
        )

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s


# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.




class GATrBlock(nn.Module):
    """Equivariant transformer block for GATr.

    This is the biggest building block of GATr.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    self-attention, and a residual connection. Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    attention: SelfAttentionConfig
        Attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        checkpoint: Optional[Sequence[Literal["mlp", "attention"]]] = None,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["mlp", "attention"]
        self._checkpoint_mlp = checkpoint is not None and "mlp" in checkpoint
        self._checkpoint_attn = checkpoint is not None and "attention" in checkpoint

        # Normalization layer (stateless, so we can use the same layer for both normalization
        # instances)
        self.norm = EquiLayerNorm()

        # Self-attention layer
        attention = replace(
            attention,
            in_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = SelfAttention(attention)

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        reference_mv: Optional[torch.Tensor] = None,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[Union[AttentionBias, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer block.

        Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
        self-attention, and a residual connection. Then the data is processed by a block consisting
        of another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and
        another residual connection.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., items, channels, 16)
            Input multivectors.
        scalars : torch.Tensor with shape (..., s_channels)
            Input scalars.
        reference_mv : torch.Tensor with shape (..., 16) or None
            Reference multivector for the equivariant join operation in the MLP.
        additional_qk_features_mv : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, multivector part.
        additional_qk_features_s : None or torch.Tensor with shape
            (..., num_items, add_qk_mv_channels, 16)
            Additional Q/K features, scalar part.
        attention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., items, channels, 16).
            Output multivectors
        output_scalars : torch.Tensor with shape (..., s_channels)
            Output scalars
        """

        # Attention block
        attn_kwargs = dict(
            multivectors=multivectors,
            scalars=scalars,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
        )
        if self._checkpoint_attn:
            h_mv, h_s = checkpoint_(self._attention_block, use_reentrant=False, **attn_kwargs)
        else:
            h_mv, h_s = self._attention_block(**attn_kwargs)

        # Skip connection
        outputs_mv = multivectors + h_mv
        outputs_s = scalars + h_s

        # MLP block
        mlp_kwargs = dict(multivectors=outputs_mv, scalars=outputs_s, reference_mv=reference_mv)
        if self._checkpoint_mlp:
            h_mv, h_s = checkpoint_(self._mlp_block, use_reentrant=False, **mlp_kwargs)
        else:
            h_mv, h_s = self._mlp_block(outputs_mv, scalars=outputs_s, reference_mv=reference_mv)

        # Skip connection
        outputs_mv = outputs_mv + h_mv
        outputs_s = outputs_s + h_s

        return outputs_mv, outputs_s

    def _attention_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        additional_qk_features_mv: Optional[torch.Tensor] = None,
        additional_qk_features_s: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.attention(
            h_mv,
            scalars=h_s,
            additional_qk_features_mv=additional_qk_features_mv,
            additional_qk_features_s=additional_qk_features_s,
            attention_mask=attention_mask,
        )
        return h_mv, h_s

    def _mlp_block(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
        reference_mv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLP block."""

        h_mv, h_s = self.norm(multivectors, scalars=scalars)
        h_mv, h_s = self.mlp(h_mv, scalars=h_s, reference_mv=reference_mv)
        return h_mv, h_s


# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All rights reserved.
"""Equivariant transformer for multivector data."""





class GATr(nn.Module):
    """GATr network for a data with a single token dimension.

    This, together with gatr.nets.axial_gatr.AxialGATr, is the main architecture proposed in our
    paper.

    It combines `num_blocks` GATr transformer blocks, each consisting of geometric self-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_blocks : int
        Number of transformer blocks.
    checkpoint_blocks : bool
        Deprecated option to specify gradient checkpointing. Use `checkpoint=["block"]` instead
    dropout_prob : float or None
        Dropout probability
    checkpoint : None or sequence of "mlp", "attention", "block"
        Which components to apply gradient checkpointing to
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        mlp: MLPConfig,
        num_blocks: int = 10,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        checkpoint: Union[
            None, Sequence[Literal["block"]], Sequence[Literal["mlp", "attention"]]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Gradient checkpointing settings
        if checkpoint_blocks:
            # The checkpoint_blocks keyword was deprecated in v1.4.0.
            if checkpoint is not None:
                raise ValueError(
                    "Both checkpoint_blocks and checkpoint were specified. Please only use"
                    "checkpoint."
                )
            warn(
                'The checkpoint_blocks keyword is deprecated since v1.4.0. Use checkpoint=["block"]'
                "instead.",
                category=DeprecationWarning,
            )
            checkpoint = ["block"]
        if checkpoint is not None:
            for key in checkpoint:
                assert key in ["block", "mlp", "attention"]
        if checkpoint is not None and "block" in checkpoint:
            self._checkpoint_blocks = True
            if "mlp" in checkpoint or "attention" in checkpoint:
                raise ValueError(
                    "Checkpointing both on the block level and the MLP / attention"
                    'level is not sensible. Please use either checkpoint=["block"] or '
                    f'checkpoint=["attention", "mlp"]. Found checkpoint={checkpoint}.'
                )
            checkpoint = None
        else:
            self._checkpoint_blocks = False

        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),  # convert duck typing to actual class
            additional_qk_mv_channels=(
                0 if reinsert_mv_channels is None else len(reinsert_mv_channels)
            ),
            additional_qk_s_channels=0 if reinsert_s_channels is None else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.blocks = nn.ModuleList(
            [
                GATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    checkpoint=checkpoint,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        join_reference: Union[Tensor, str] = "data",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        attention_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask
        join_reference : Tensor with shape (..., 16) or {"data", "canonical"}
            Reference multivector for the equivariant joint operation. If "data", a
            reference multivector is constructed from the mean of the input multivectors. If
            "canonical", a constant canonical reference multivector is used instead.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """

        # Reference multivector and channels that will be re-inserted in any query / key computation
        reference_mv = construct_reference_multivector(join_reference, multivectors)
        additional_qk_features_mv, additional_qk_features_s = self._construct_reinserted_channels(
            multivectors, scalars
        )

        # Pass through the blocks
        h_mv, h_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.blocks:
            if self._checkpoint_blocks:
                h_mv, h_s = checkpoint_(
                    block,
                    h_mv,
                    use_reentrant=False,
                    scalars=h_s,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )
            else:
                h_mv, h_s = block(
                    h_mv,
                    scalars=h_s,
                    reference_mv=reference_mv,
                    additional_qk_features_mv=additional_qk_features_mv,
                    additional_qk_features_s=additional_qk_features_s,
                    attention_mask=attention_mask,
                )

        outputs_mv, outputs_s = self.linear_out(h_mv, scalars=h_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s




__all__ = [
    "GATr",
    "GATrBlock",
    "SelfAttentionConfig",
    "MLPConfig",
    "EquiLinear",
    "SelfAttention",
    "GeoMLP",
    "embed_point",
    "embed_translation",
]
