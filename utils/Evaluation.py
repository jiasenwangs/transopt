# -*- coding: utf-8 -*-
"""
@author: Jiasen Wang
"""
import logging

def MAE(targets,preds):
    samples = len(targets)
    errors = []
    for i in range(samples):
        a = targets[i] 
        b = preds[i]
        
        error = [ abs(a[j]-b[j]) for j in range(len(a))]
        errors.append(sum(error)/len(error) )
    return 1.0*sum(errors)/samples