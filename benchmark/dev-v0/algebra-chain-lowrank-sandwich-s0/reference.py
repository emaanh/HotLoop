import torch


def reference(a, b, c, d):
    """a: (d0, d1), b: (d1, d2), c: (d2, d3), d: (d3, d4) -> (d0, d4)"""
    return a @ b @ c @ d
