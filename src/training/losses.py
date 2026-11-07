"""Training losses. Each one uses only pixels with valid ground truth."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.depthguard_net import DepthGuardOutput

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the mean over masked pixels, or a differentiable zero if none are valid."""
    if not mask.any():
        return values.sum() * 0
    return values[mask].mean()


def laplace_nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    mask: torch.Tensor,
    reference_m: float,
) -> torch.Tensor:
    """Return the Laplace negative log-likelihood, with scale b = reference * exp(log_variance).

    Constants are dropped. This one term trains both depth and confidence.
    """
    scale = reference_m * torch.exp(log_variance)
    return _masked_mean(torch.abs(prediction - target) / scale + log_variance, mask)


def sobel_gradients(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal and vertical Sobel gradients in metres per pixel."""
    kernel_x = _SOBEL_X.to(depth.device, depth.dtype)[None, None]
    return F.conv2d(depth, kernel_x, padding=1), F.conv2d(depth, kernel_x.transpose(2, 3), padding=1)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return L1 between Sobel gradients, where the whole 3x3 window is valid.

    The image's outermost pixels are excluded, since their windows are incomplete.
    """
    pred_x, pred_y = sobel_gradients(prediction)
    target_x, target_y = sobel_gradients(target)
    window_valid = -F.max_pool2d(-mask.to(prediction.dtype), 3, stride=1, padding=1) > 0.5
    window_valid = window_valid.clone()
    window_valid[..., 0, :] = window_valid[..., -1, :] = False
    window_valid[..., :, 0] = window_valid[..., :, -1] = False
    return _masked_mean(torch.abs(pred_x - target_x) + torch.abs(pred_y - target_y), window_valid)


def boundary_labels(target: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
    """Build boolean labels from ground truth depth, matching ``src.utils.depth_edges``.

    A pixel is a boundary if the relative jump to its right or lower neighbour exceeds
    ``threshold`` and both pixels are valid. RGB is not used.
    """
    usable = mask & (target > 0)
    labels = torch.zeros_like(mask)
    pair_x = usable[..., :, :-1] & usable[..., :, 1:]
    jump_x = torch.abs(target[..., :, :-1] - target[..., :, 1:]) / torch.minimum(
        target[..., :, :-1], target[..., :, 1:]
    ).clamp_min(1e-6)
    labels[..., :, :-1] |= pair_x & (jump_x > threshold)
    pair_y = usable[..., :-1, :] & usable[..., 1:, :]
    jump_y = torch.abs(target[..., :-1, :] - target[..., 1:, :]) / torch.minimum(
        target[..., :-1, :], target[..., 1:, :]
    ).clamp_min(1e-6)
    labels[..., :-1, :] |= pair_y & (jump_y > threshold)
    return labels


def boundary_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return BCE over valid pixels, with positives up-weighted since boundaries are rare."""
    positives = (labels & mask).sum().clamp_min(1)
    negatives = (~labels & mask).sum().clamp_min(1)
    pos_weight = (negatives / positives).clamp_max(20.0).to(logits.dtype)
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, labels.to(logits.dtype), pos_weight=pos_weight, reduction="none"
    )
    return _masked_mean(per_pixel, mask)


class DepthGuardLoss(nn.Module):
    """Weighted sum of the depth, gradient and boundary losses."""

    def __init__(self, weights: dict[str, float], reference_m: float, boundary_threshold: float) -> None:
        """Store the loss settings from the ``train`` section of the config."""
        super().__init__()
        self.weights = weights
        self.reference_m = reference_m
        self.boundary_threshold = boundary_threshold

    def forward(
        self,
        output: DepthGuardOutput,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return each unweighted loss term and the weighted total."""
        terms = {
            "depth": laplace_nll_loss(output.depth, target, output.log_variance, mask, self.reference_m),
            "gradient": gradient_loss(output.depth, target, mask),
            "boundary": boundary_loss(
                output.boundary_logits, boundary_labels(target, mask, self.boundary_threshold), mask
            ),
        }
        terms["total"] = sum(self.weights[name] * value for name, value in terms.items())
        return terms
