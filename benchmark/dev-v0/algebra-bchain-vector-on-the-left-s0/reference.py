import torch


def reference(x, w1, w2, y):
    """x: (B, d0, d1) per-sample, w1: (d1, d2) shared, w2: (d2, d3) shared, y: (B, d3, d4)
    per-sample -> (B, d0, d4)"""
    return x @ w1 @ w2 @ y
