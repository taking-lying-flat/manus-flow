from dataclasses import dataclass, field

from torch import Tensor

from .utils import append_dims


@dataclass
class PathSample:
    x_t: Tensor = field(metadata={"help": "x_t = noise.lerp(data, t)"})
    dx_t: Tensor = field(metadata={"help": "flow target: data - noise"})
    noised: Tensor = field(metadata={"help": "alias for x_t"})
    flow: Tensor = field(metadata={"help": "alias for dx_t"})


class RectifiedFlowPath:
    """Linear interpolation path: x_t = (1-t)*noise + t*data, flow = data - noise."""

    def sample(self, noise: Tensor, data: Tensor, t: Tensor) -> PathSample:
        padded_t = append_dims(t, data.ndim - 1)
        x_t = noise.lerp(data, padded_t)
        dx_t = data - noise
        return PathSample(x_t=x_t, dx_t=dx_t, noised=x_t, flow=dx_t)
