from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class EMAHelper:
    def __init__(self, mu: float = 0.999, *, use_buffers: bool = True) -> None:
        self.decay = mu
        self.use_buffers = use_buffers
        self._model: AveragedModel | None = None

    @property
    def averaged_model(self) -> AveragedModel:
        if self._model is None:
            raise RuntimeError("EMAHelper.register(module) must be called before using EMA state.")
        return self._model

    def register(self, module: nn.Module) -> None:
        self._model = AveragedModel(
            module,
            multi_avg_fn=get_ema_multi_avg_fn(self.decay),
            use_buffers=self.use_buffers,
        )
        self._model.update_parameters(module)

    def update(self, module: nn.Module) -> None:
        self.averaged_model.update_parameters(module)

    def ema(self, module: nn.Module) -> None:
        module.load_state_dict(self.averaged_model.module.state_dict())

    def ema_copy(self, module: nn.Module) -> nn.Module:
        model_copy = copy.deepcopy(module)
        self.ema(model_copy)
        return model_copy

    def state_dict(self) -> dict[str, Any]:
        return dict(self.averaged_model.state_dict())

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.averaged_model.load_state_dict(state_dict, strict=True)
