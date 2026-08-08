# -*- coding: utf-8 -*-
"""ATFlash: per-RoPE-wavelength attention windows.

Two modules, and between them they are the whole method:

    window_table  the rule. Given a model's RoPE base, head dimension and the
                  number of retained periods k, it produces the per-pair window
                  widths and the prefix length D_eff each key tile needs. Nothing
                  here is learned or tuned; it is arithmetic on model constants.

    reference     a readable torch implementation of the windowed attention, in
                  fp32 (correctness baseline) or bf16 (error anchor for a real
                  kernel).

To put the window into your own kernel you need the table from `window_table`
and the acceptance test in `gates/`: with the window off your kernel must stay
bit-identical to its unwindowed self. Worked examples of that insertion, against
two different released kernels, are under `kernels/`.
"""
from . import reference, window_table

__all__ = ["window_table", "reference"]
