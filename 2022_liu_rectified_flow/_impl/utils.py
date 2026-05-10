from typing import Optional, TypeVar

import torch
from torch import Tensor

T = TypeVar("T")


def exists(v: Optional[T]) -> bool:
    return v is not None


def default(v: Optional[T], d: T) -> T:
    return v if exists(v) else d


def identity(t):
    return t


def append_dims(t: Tensor, ndims: int) -> Tensor:
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))


def normalize_to_neg_one_to_one(img: Tensor) -> Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(t: Tensor) -> Tensor:
    return (t + 1) * 0.5
