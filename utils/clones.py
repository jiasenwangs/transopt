# -*- coding: utf-8 -*-
"""
@author: Jiasen 
    Modified from original implementation by Harvard elites:
    https://nlp.seas.harvard.edu/annotated-transformer/
"""
import torch.nn as nn
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])