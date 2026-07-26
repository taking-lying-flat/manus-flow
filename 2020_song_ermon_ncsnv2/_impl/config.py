from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True, slots=True)
class DataConfig:
    image_size: int
    channels: int
    num_workers: int = 8
    logit_transform: bool = False
    rescaled: bool = False


@dataclass(frozen=True, slots=True)
class ModelConfig:
    ngf: int
    num_noise_levels: int
    sigma_begin: float
    sigma_end: float
    nonlinearity: str
    normalization: str
    ema: bool = False
    ema_rate: float = 0.999
    variant: str = "base"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int = 128
    anneal_power: float = 2.0
    n_iters: int = 100_000
    sample_steps_each: int = 5


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    data: DataConfig
    model: ModelConfig
    device: torch.device


EXPERIMENT_CONFIG = ExperimentConfig(
    data=DataConfig(
        image_size=32,
        channels=3,
    ),
    model=ModelConfig(
        ngf=128,
        num_noise_levels=232,
        sigma_begin=50,
        sigma_end=0.01,
        nonlinearity="elu",
        normalization="InstanceNorm++",
        ema=True,
        ema_rate=0.999,
    ),
    training=TrainingConfig(sample_steps_each=5),
)


def build_runtime_config(
    experiment: ExperimentConfig,
    *,
    device: torch.device,
    variant: str | None,
) -> RuntimeConfig:
    model = (
        experiment.model
        if variant is None
        else replace(experiment.model, variant=variant)
    )
    return RuntimeConfig(data=experiment.data, model=model, device=device)
