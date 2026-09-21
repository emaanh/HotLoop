import torch

DIMS = [8192, 64, 8192, 64, 16]


def make_inputs(entry, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        torch.randn(DIMS[i], DIMS[i + 1], device=device, generator=g) / DIMS[i + 1] ** 0.5
        for i in range(4)
    )
