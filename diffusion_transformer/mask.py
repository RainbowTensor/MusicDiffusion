import torch

def scalar_to_batch_tensor(x, batch_size):
    return torch.tensor(x).repeat(batch_size)

def _gamma(r):
    return (r * torch.pi / 2).cos().clamp(1e-10, 1.0)

def generate_mask(
    x: torch.Tensor,
    r: torch.Tensor
):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device)

    r = _gamma(r)[:, None, None]
    probs = torch.ones_like(x) * r

    mask = torch.bernoulli(probs)
    mask = mask.round().long()

    return mask

def apply_mask(
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: int
    ):
    assert mask.ndim == 3, "mask must be (batch, n_codebooks, seq), but got {mask.ndim}"
    assert mask.shape == x.shape, f"mask must be same shape as x, but got {mask.shape} and {x.shape}"
    assert mask.dtype == torch.long, "mask must be long dtype, but got {mask.dtype}"
    assert ~torch.any(mask > 1), "mask must be binary"
    assert ~torch.any(mask < 0), "mask must be binary"

    fill_x = torch.full_like(x, mask_token)
    x = x * (1 - mask) + fill_x * mask

    return x, mask