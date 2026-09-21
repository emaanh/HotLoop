import torch

BATCH = 64
DIMS = [256, 64, 2048, 64, 256]


def make_inputs(entry, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)

    def m(*shape):
        return torch.randn(*shape, device=device, generator=g) / shape[-1] ** 0.5

    d = DIMS
    return (m(BATCH, d[0], d[1]), m(d[1], d[2]), m(d[2], d[3]), m(BATCH, d[3], d[4]))
