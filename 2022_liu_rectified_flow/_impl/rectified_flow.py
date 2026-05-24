from typing import Callable, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange, repeat

from .path import RectifiedFlowPath
from .solver import ODESolver
from .loss import LossBreakdown, build_loss_fn
from .timestep_sampler import build_timestep_sampler
from .utils import (
    append_dims,
    default,
    exists,
    identity,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

from .unet import Unet


class EMA(nn.Module):
    """EMA wrapper that tracks an online model's parameters."""

    def __init__(
        self,
        online_model: nn.Module,
        beta: float = 0.9999,
        update_after_step: int = 100,
    ):
        super().__init__()
        self.beta = beta
        self.update_after_step = update_after_step
        self.online_model = online_model

        from copy import deepcopy

        self.ema_model = deepcopy(online_model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.step = 0

    def update(self):
        self.step += 1
        if self.step < self.update_after_step:
            self._copy_direct()
            return

        with torch.no_grad():
            for ema_p, online_p in zip(
                self.ema_model.parameters(), self.online_model.parameters()
            ):
                ema_p.lerp_(online_p, 1.0 - self.beta)
            for ema_b, online_b in zip(
                self.ema_model.buffers(), self.online_model.buffers()
            ):
                ema_b.copy_(online_b)

    def _copy_direct(self):
        """Direct copy before warmup steps complete."""
        for ema_p, online_p in zip(
            self.ema_model.parameters(), self.online_model.parameters()
        ):
            ema_p.copy_(online_p)
        for ema_b, online_b in zip(
            self.ema_model.buffers(), self.online_model.buffers()
        ):
            ema_b.copy_(online_b)

    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        model: dict | nn.Module,
        time_cond_kwarg: str | None = "times",
        odeint_kwargs: dict | None = None,
        predict: Literal["flow", "noise", "clean"] = "flow",
        loss_fn: Literal["mse", "pseudo_huber", "pseudo_huber_with_lpips"] | str = "mse",
        timestep_sampler: Literal["uniform", "logit_normal", "cosmap", "mode"] | str = "uniform",
        timestep_sampler_kwargs: dict | None = None,
        ema_update_after_step: int = 100,
        data_shape: Optional[Tuple[int, ...]] = None,
        use_consistency: bool = False,
        max_timesteps: int = 100,
        consistency_decay: float = 0.9999,
        consistency_velocity_match_alpha: float = 1e-5,
        consistency_delta_time: float = 1e-3,
        consistency_loss_weight: float = 1.0,
        data_normalize_fn: Callable = normalize_to_neg_one_to_one,
        data_unnormalize_fn: Callable = unnormalize_to_zero_to_one,
        clip_during_sampling: bool = False,
        clip_values: Tuple[float, float] = (-1.0, 1.0),
        clip_flow_during_sampling: Optional[bool] = None,
        clip_flow_values: Tuple[float, float] = (-3.0, 3.0),
        eps: float = 5e-3,
    ):
        super().__init__()

        if isinstance(model, dict):
            model = Unet(**model)

        self.model = model
        self.time_cond_kwarg = time_cond_kwarg
        self.predict = predict
        self.max_timesteps = max_timesteps

        clip_flow_during_sampling = default(
            clip_flow_during_sampling, predict == "noise"
        )

        # loss
        self.loss_fn = build_loss_fn(loss_fn)

        # timestep sampler controls which t values are sampled more frequently
        timestep_sampler_kwargs = default(timestep_sampler_kwargs, {})
        self.timestep_sampler = build_timestep_sampler(
            timestep_sampler, **timestep_sampler_kwargs
        )

        # path always uses linear interpolation: x_t = (1-t)*noise + t*data
        self.path = RectifiedFlowPath()

        # solver
        self.solver = ODESolver(odeint_kwargs=odeint_kwargs)

        self.data_shape = data_shape

        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling
        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # consistency
        self.use_consistency = use_consistency
        self.consistency_delta_time = consistency_delta_time
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                online_model=self.model,
                beta=consistency_decay,
                update_after_step=ema_update_after_step,
            )

        self.data_normalize_fn = data_normalize_fn
        self.data_unnormalize_fn = data_unnormalize_fn

        self.eps = eps

    def post_training_step_update(self):
        if self.use_consistency:
            self.ema_model.update()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model: nn.Module, noised, *, times, **model_kwargs):
        batch = noised.shape[0]
        time_kwarg = self.time_cond_kwarg

        if exists(time_kwarg):
            times = rearrange(times, "... -> (...)")
            if times.numel() == 1:
                times = repeat(times, "1 -> b", b=batch)
            model_kwargs.update(**{time_kwarg: times})

        output = model(noised, **model_kwargs)

        if self.predict == "flow":
            flow = output
        elif self.predict == "noise":
            noise_pred = output
            padded_times = append_dims(times, noised.ndim - 1)
            flow = (noised - noise_pred) / padded_times.clamp_min(self.eps)
        elif self.predict == "clean":
            clean = output
            padded_times = append_dims(times, noised.ndim - 1)
            scale = 1.0 / (1.0 - padded_times).clamp_min(self.eps)
            flow = (clean - noised) * scale
        else:
            raise ValueError(f"unknown objective {self.predict}")

        return output, flow

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        steps: int = 16,
        noise: Optional[Tensor] = None,
        data_shape: Optional[Tuple[int, ...]] = None,
        use_ema: bool = False,
        **model_kwargs,
    ):
        use_ema = default(use_ema, self.use_consistency)
        if use_ema and not self.use_consistency:
            raise ValueError(
                "use_ema=True requires use_consistency=True during init"
            )

        model = self.ema_model if use_ema else self.model

        was_training = self.training
        self.eval()

        data_shape = default(data_shape, self.data_shape)
        if data_shape is None:
            raise ValueError(
                "data_shape must be provided or set during training"
            )

        maybe_clip = (
            (lambda t: t.clamp(*self.clip_values))
            if self.clip_during_sampling
            else identity
        )
        maybe_clip_flow = (
            (lambda t: t.clamp(*self.clip_flow_values))
            if self.clip_flow_during_sampling
            else identity
        )

        def ode_fn(t, x):
            x = maybe_clip(x)
            _, flow = self.predict_flow(model, x, times=t, **model_kwargs)
            return maybe_clip_flow(flow)

        noise = default(
            noise, torch.randn((batch_size, *data_shape), device=self.device)
        )

        sampled_data = self.solver.sample(ode_fn=ode_fn, x_init=noise, steps=steps)

        self.train(was_training)
        return self.data_unnormalize_fn(sampled_data)

    def forward(
        self,
        data,
        noise: Optional[Tensor] = None,
        return_loss_breakdown: bool = False,
        **model_kwargs,
    ):
        batch, *data_shape = data.shape
        data = self.data_normalize_fn(data)
        self.data_shape = default(self.data_shape, data_shape)

        noise = default(noise, torch.randn_like(data))

        times = self.timestep_sampler.sample(batch, self.device)

        if self.predict == "clean":
            # Avoid t=1 where flow = (clean - noised) / (1-t) blows up.
            # At sampling time the ODE solver will query t=1 — the model
            # must extrapolate over the gap of 1/max_timesteps (≈0.01).
            times = times * (1.0 - self.max_timesteps**-1)

        if self.use_consistency:
            times *= 1.0 - self.consistency_delta_time

        def get_noised_and_flows(model, t):
            path_sample = self.path.sample(noise=noise, data=data, t=t)
            noised, flow = path_sample.noised, path_sample.flow
            padded_t = append_dims(t, noised.ndim - 1)

            model_output, pred_flow = self.predict_flow(
                model, noised, times=t, **model_kwargs
            )

            # pred_data = noised + pred_flow * (1 - t)  (RF one-step prediction)
            pred_data = noised + pred_flow * (1.0 - padded_t)

            return model_output, flow, pred_flow, pred_data

        output, flow, pred_flow, pred_data = get_noised_and_flows(
            self.model, times
        )

        if self.use_consistency:
            delta_t = self.consistency_delta_time
            with torch.no_grad():
                self.ema_model.eval()
                ema_output, ema_flow, ema_pred_flow, ema_pred_data = (
                    get_noised_and_flows(self.ema_model, times + delta_t)
                )

        if self.predict == "flow":
            pred, target = output, flow
        elif self.predict == "noise":
            pred, target = output, noise
        elif self.predict == "clean":
            pred, target = pred_flow, flow
        else:
            raise ValueError(f"unknown objective {self.predict}")

        main_loss = self.loss_fn(
            pred, target, pred_data=pred_data, times=times, data=data
        )

        consistency_loss = data_match_loss = velocity_match_loss = 0.0

        if self.use_consistency:
            data_match_loss = F.mse_loss(pred_data, ema_pred_data)
            velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)
            consistency_loss = (
                data_match_loss
                + velocity_match_loss * self.consistency_velocity_match_alpha
            )

        total_loss = main_loss + consistency_loss * self.consistency_loss_weight

        if not return_loss_breakdown:
            return total_loss

        return total_loss, LossBreakdown(
            total_loss, main_loss, data_match_loss, velocity_match_loss
        )
