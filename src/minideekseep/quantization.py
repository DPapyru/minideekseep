"""Software fake quantization for QAT studies; this does not accelerate SM75 kernels."""

import torch
import torch.nn.functional as F


def fake_fp4(x, group_size=16):
    original = x.shape
    padding = (-x.size(-1)) % group_size
    groups = F.pad(x.float(), (0, padding)).unflatten(-1, (-1, group_size))
    scale = (groups.abs().amax(-1, keepdim=True) / 6).clamp_min(2**-9)
    scale = scale.clamp(max=448).to(torch.float8_e4m3fn).float().clamp_min(2**-9)
    codebook = x.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6]).float()
    normalized = groups.abs() / scale
    indices = (normalized.unsqueeze(-1) - codebook).abs().argmin(-1)
    quantized = codebook[indices] * groups.sign() * scale
    quantized = quantized.flatten(-2)[..., : original[-1]].to(x.dtype)
    return x + (quantized - x).detach()


def fake_fp8(x):
    scale = (x.detach().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-9)
    quantized = ((x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).to(x.dtype)
    return x + (quantized - x).detach()
