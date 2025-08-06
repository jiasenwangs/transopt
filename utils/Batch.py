# -*- coding: utf-8 -*-
"""
@author: Jiasen 
    reference: https://nlp.seas.harvard.edu/annotated-transformer/
"""
from utils.subsequent_mask import subsequent_mask

class Batch:
    "This batch is for PPO sampling"
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, info, batch_max_len, remain_bits, pad=2):  # 2 = <blank>
        self.src = src
        # self.src_mask = (src[:,:] != pad).unsqueeze(-2)
        self.src_mask = (src != pad).unsqueeze(-2) #TODO check
        self.info = info # store info to compute function values and decoding
        self.batch_max_len = batch_max_len
        self.remain_bits = remain_bits
        
    def to(self, device):
        # explicit move tensor to device, for usage when num_workers>0
        self.src = self.src.to(device,non_blocking=True)
        self.src_mask = self.src_mask.to(device,non_blocking=True)
        # self.tgt = self.tgt.to(device,non_blocking=True)
        # self.tgt_y = self.tgt_y.to(device,non_blocking=True)
        # self.tgt_mask = self.tgt_mask.to(device,non_blocking=True)
        self.remain_bits = self.remain_bits.to(device,non_blocking=True)

class BatchOffline:
    "This batch is for PPO sampling wit guidance"
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, tgt, src_pad, tgt_pad):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != src_pad).unsqueeze(-2) 
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]
        self.tgt_mask = make_std_mask(self.tgt, tgt_pad)
        self.ntokens = (self.tgt_y != tgt_pad).data.sum()
    
def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
        tgt_mask.data
    )
    return tgt_mask