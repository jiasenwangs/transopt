# -*- coding: utf-8 -*-
"""
@author: Jiasen 
    reference: https://nlp.seas.harvard.edu/annotated-transformer/
"""
# import torch
import numpy as np

class LossCompute:
    "A loss compute supervised training loss KL divergence."
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator.logforward(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss