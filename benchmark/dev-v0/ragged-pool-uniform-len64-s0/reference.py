import torch

LMAX = 64  # no segment is longer than this


def reference(v, s, offsets):
    """v: (T, D) values, s: (T,) scores, offsets: (S+1,) int64 segment boundaries into T.
    Returns (S, D): for each segment, sum_i softmax(s_seg)_i * v_seg_i."""
    lens = offsets[1:] - offsets[:-1]
    idx = torch.arange(LMAX, device=v.device)[None, :]
    mask = idx < lens[:, None]
    pos = (offsets[:-1, None] + idx).clamp_max(v.shape[0] - 1)
    sp = s[pos].masked_fill(~mask, float("-inf"))
    w = torch.softmax(sp.float(), dim=-1).to(v.dtype)
    return (w[..., None] * v[pos]).sum(1)
