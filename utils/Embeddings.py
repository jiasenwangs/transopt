# -*- coding: utf-8 -*-
"""
@author: Jiasen 
    Copied from original implementation by Harvard elites:
    https://nlp.seas.harvard.edu/annotated-transformer/
"""
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
        # return self.lut(x) # seems both are OK, I use the above vanilla version