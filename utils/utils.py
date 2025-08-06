# -*- coding: utf-8 -*-
"""
@author: Jiasen 
    Modified from original implementation:
    https://nlp.seas.harvard.edu/annotated-transformer/
"""

import torch
import numpy as np
from functools import partial
# import pandas as pd
import ast
# import time
import logging
import copy
import json
import torch.nn as nn

# from .subsequent_mask import subsequent_mask
from .EncoderDecoder import EncoderDecoder,Encoder,EncoderLayer,Decoder,DecoderLayer
from .Generator import Generator
from .MultiHeadedAttention import MultiHeadedAttention
from .PositionwiseFeedForward import PositionwiseFeedForward
from .Embeddings import Embeddings
from .PositionalEncoding import PositionalEncoding


def make_model(src_vocab, tgt_vocab, N=6, d_model=32, d_ff=50, h=8, dropout=0.0):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, max_len = 1000)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def getFunc(func_str):
    d = {}
    exec(func_str, d)
    func = next(d[k] for k in d if k != '__builtins__')
    return func

def decode(txts,info):
    # ys = []
    xs = []
    for i in range(len(txts)):
        txt = txts[i]
        
        info_i = json.loads(info[i] )
        solu_bits = info_i["solu_bits"]
        f = getFunc(info_i["f"])
        g = getFunc(info_i["g"])
        h = getFunc(info_i["h"])
        parameter = info_i["parameter"]
        x = ast.literal_eval(txt)
        xs.append(x)
    return  xs

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    mode="train"
):
    """Train a single epoch"""
    total_loss = 0
    tokens = 0 
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train":
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss
        tokens += batch.ntokens
        del loss
        del loss_node
        
    return total_loss,tokens

def run_model_test(test_dataloader,
                   model,
                   pad_idx,
                   eos_string,
                   start_id,
                   cuda,
                   device,
                   tgt_vocab,
                   args):
    logging.info("Checking Model Outputs:")
    targets,preds = check_outputs(
        test_dataloader, 
        model,
        args
    )
    
    return targets,preds

def check_outputs(
    data_iter,
    model,
    args=None
):
    # results = [()] * n_examples
    targets = []
    preds = []
    for idx, rb in enumerate(data_iter):
        logging.info("Batch %d ==========" % idx)
        with torch.no_grad():
            txts = model.module.inference(rb.src,
                            rb.src_mask,
                            args.sos_token,
                            args.pad_token,
                            args.enc,
                            max_len=args.max_output_len, 
                            sos_string=args.sos_char,
                            eos_string=args.eos_char,
                            deterministic=True)
        logging.info("Program info:")
        logging.info(f"{rb.info}")
        
        logging.info("Model output txts:")
        logging.info(txts)
        xs = decode(txts, rb.info)
        y_star_list = []
        x_star_list = []
        for i in range(len(txts)):
            # txt_i = txts[i]
            info_i = json.loads(rb.info[i] )
            
            func_star = getFunc(info_i["func_star"])
            parameter = info_i["parameter"]
            f = getFunc(info_i["f"])
            x_star,y_star = func_star(parameter,np,f)
            

            logging.info("Ground truth objective function value:")
            logging.info(y_star)
            logging.info("Ground truth program solution:")
            logging.info(x_star)
            logging.info("Model output program solution:")
            logging.info(xs[i])
            y_star_list.append(y_star)
            x_star_list.append(x_star)
        targets.extend(x_star_list)
        preds.extend(xs)
    return targets, preds

def reduce_tensor(tensor,dist):
    rt = tensor.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = torch.sum(rt[0:-1])/rt[-1]
    return rt