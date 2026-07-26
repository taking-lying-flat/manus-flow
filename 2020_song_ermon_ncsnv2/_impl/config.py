from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True, slots=True)
class DataConfig:
    dataset: str
    image_size: int
    channels: int
    category: str | None = None
    random_flip: bool = False
    num_workers: int = 8
    logit_transform: bool = False
    uniform_dequantization: bool = False
    gaussian_dequantization: bool = False
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


EXPERIMENT_CONFIGS: dict[str, ExperimentConfig] = {
    "cifar10": ExperimentConfig(
        data=DataConfig(
            dataset="cifar10",
            image_size=32,
            channels=3,
            random_flip=True,
            uniform_dequantization=True,
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
    ),
    "celeba": ExperimentConfig(
        data=DataConfig(
            dataset="celeba",
            image_size=64,
            channels=3,
            random_flip=True,
            num_workers=32,
        ),
        model=ModelConfig(
            ngf=128,
            num_noise_levels=500,
            sigma_begin=90.0,
            sigma_end=0.01,
            nonlinearity="elu",
            normalization="InstanceNorm++",
            ema=True,
        ),
        training=TrainingConfig(n_iters=200_001),
    ),
}


def build_runtime_config(
    experiment: ExperimentConfig,
    *,
    device: torch.device,
    variant: str | None,
) -> RuntimeConfig:
    model = experiment.model if variant is None else replace(experiment.model, variant=variant)
    return RuntimeConfig(data=experiment.data, model=model, device=device)
