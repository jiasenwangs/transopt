# -*- coding: utf-8 -*-
"""
@author: Jiasen 
    Copied from original implementation by Harvard elites:
    https://nlp.seas.harvard.edu/annotated-transformer/
"""
import torch

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0