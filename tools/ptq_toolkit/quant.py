import torch


def quantize_symmetric(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    """Reference symmetric uniform quantizer."""
    if bits < 2:
        raise ValueError(f"bits must be >= 2, got {bits}")

    qmax = (1 << (bits - 1)) - 1
    alpha = x.detach().abs().max()
    scale = torch.clamp(alpha / qmax, min=eps)
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return q * scale

