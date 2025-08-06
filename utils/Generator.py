# -*- coding: utf-8 -*-
"""
@author: Jiasen 
    Copied from original implementation by Harvard elites:
    https://nlp.seas.harvard.edu/annotated-transformer/
"""

import torch.nn as nn
from torch.nn.functional import log_softmax
# import torch.nn.functional as F

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, mask):
        proj = self.proj(x)
        if mask is not None:
            proj = proj.masked_fill(mask == 0, -1e9)
        # return proj.softmax(dim=-1)
        return log_softmax(proj, dim=-1)
    
    def logforward(self, x):
        "Note: no mask"
        return log_softmax(self.proj(x), dim=-1)
    