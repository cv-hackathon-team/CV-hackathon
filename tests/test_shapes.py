"""Shape tests for DepthGuard360. Run with ``python -m pytest tests/test_shapes.py -s`` to see the printout."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.classical.guided_filter import guided_upsample
from src.classical.interpolation import upsample_bicubic, upsample_bilinear, upsample_nearest
from src.classical.joint_bilateral import joint_bilateral_upsample
from src.datasets.corruption import MODES, corrupt_depth, make_rng
from src.models.depthguard_net import DepthGuardNet, count_parameters
from src.utils import load_config

SCALE_FACTORS = [2, 4, 8]


def _scene(height: int = 64, width: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Return a two-plane depth map with a sharp edge and a matching RGB guide."""
    depth = np.full((height, width), 2.0, dtype=np.float32)
    depth[:, width // 2 :] = 4.0
    rgb = np.full((height, width, 3), 0.2, dtype=np.float32)
    rgb[:, width // 2 :] = 0.8
    return depth, rgb


def _describe(value: object) -> str:
    """Return a readable shape for a tensor, a list of tensors or a network output."""
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    if isinstance(value, (list, tuple)):
        return " ".join(_describe(item) for item in value)
    return " ".join(f"{k}={_describe(v)}" for k, v in vars(value).items())


def test_network_params_and_shapes_at_each_stage() -> None:
    """The network stays under 5M parameters, and all heads return (B, 1, H, W)."""
    config = load_config()["model"]
    model = DepthGuardNet(config["channels"], config["blocks_per_level"])
    params = count_parameters(model)
    print(f"\nTotal parameters: {params:,}")
    for name, child in model.named_children():
        print(f"  {name:18s} {count_parameters(child):>10,}")
    assert params < 5_000_000

    def report(name: str):
        return lambda _, __, output: print(f"  {name:18s} -> {_describe(output)}")

    handles = [model.rgb_encoder.register_forward_hook(report("rgb_encoder")),
               model.depth_encoder.register_forward_hook(report("depth_encoder")),
               model.decoder.register_forward_hook(report("decoder"))]
    handles += [block.register_forward_hook(report(f"fusion[{i}]")) for i, block in enumerate(model.fusion)]
    handles += [getattr(model, f"{head}_head").register_forward_hook(report(f"{head}_head"))
                for head in ("depth", "log_variance", "boundary")]
    rgb, depth, mask = torch.rand(2, 3, 256, 256), torch.rand(2, 1, 256, 256) + 1, torch.ones(2, 1, 256, 256, dtype=torch.bool)
    print(f"  inputs             -> rgb={_describe(rgb)} depth={_describe(depth)} mask={_describe(mask)}")
    output = model(rgb, depth, mask)
    print(f"  output             -> {_describe(output)}")
    for handle in handles:
        handle.remove()
    for tensor in (output.depth, output.log_variance, output.confidence, output.boundary_logits):
        assert tensor.shape == (2, 1, 256, 256)
    assert (output.confidence > 0).all() and (output.confidence <= 1).all()


def test_network_starts_as_identity_on_depth() -> None:
    """Before training, the depth output equals the bicubic input."""
    model = DepthGuardNet().eval()
    depth = torch.rand(1, 1, 64, 64) + 1
    output = model(torch.rand(1, 3, 64, 64), depth, torch.ones(1, 1, 64, 64, dtype=torch.bool))
    assert torch.allclose(output.depth, depth)


def test_network_rejects_bad_size() -> None:
    """A size not divisible by 8 fails with a clear error."""
    with pytest.raises(ValueError, match="divisible"):
        DepthGuardNet()(torch.rand(1, 3, 60, 60), torch.rand(1, 1, 60, 60), torch.ones(1, 1, 60, 60, dtype=torch.bool))


@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
def test_classical_upsampling_shapes(scale_factor: int) -> None:
    """Classical methods return the target resolution."""
    depth, rgb = _scene()
    lr = depth[:: scale_factor, :: scale_factor]
    valid = np.ones_like(lr, dtype=bool)
    outputs = [
        upsample_nearest(lr, scale_factor, valid),
        upsample_bilinear(lr, scale_factor, valid),
        upsample_bicubic(lr, scale_factor, valid),
        joint_bilateral_upsample(lr, rgb, scale_factor, validity_mask=valid),
        guided_upsample(lr, rgb, scale_factor, validity_mask=valid),
    ]
    assert all(output.shape == depth.shape for output in outputs)


@pytest.mark.parametrize("scale_factor", SCALE_FACTORS)
@pytest.mark.parametrize("mode", MODES)
def test_corruption_shapes(mode: str, scale_factor: int) -> None:
    """Each corruption mode returns a depth map and a matching boolean mask."""
    depth, _ = _scene()
    params = load_config()["corruption"]["levels"]["medium"]
    result = corrupt_depth(
        depth, np.ones_like(depth, dtype=bool), scale_factor, mode, params, make_rng(1, 0, scale_factor, mode)
    )
    expected = (depth.shape[0] // scale_factor, depth.shape[1] // scale_factor)
    assert result.lr_depth.shape == expected
    assert result.validity_mask.shape == expected
    assert result.validity_mask.dtype == bool
    assert result.label == mode
