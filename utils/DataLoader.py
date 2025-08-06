"""
@author: Jiasen 
    Modified from original implementation:
    https://github.com/gentaiscool/end2end-asr-pytorch
"""

import numpy as np
import pandas as pd
# import ast
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import inspect
import json
# from .Binary import float_to_bin
# from .Binary import float2bin_half

from .FunctionsTest import obj1,h1,g1,obj1_optimal
from .FunctionsTest import obj2,h2,g2,obj2_optimal
from .FunctionsTest import obj3,h3,g3,obj3_optimal
from .FunctionsTest import obj4,h4,g4,obj4_optimal
from .FunctionsTest import obj5,h5,g5,obj5_optimal
from .FunctionsTest import obj6,h6,g6,obj6_optimal
from .FunctionsTest import obj7,h7,g7,obj7_optimal
from .FunctionsTest import obj8,h8,g8,obj8_optimal
from .FunctionsTest import cons_basic

class ProblemDataset(Dataset):
    def __init__(self,
                 max_size=100, # maximum number of samples for an epoch
                 src_max_len=4000,
                 sos_token=0,
                 eos_token=1,
                 is_train=True,
                 local_search = False):
        self.src_max_len = src_max_len
        
        self.max_size = max_size
        # self.max_size = len(self.ids_list)
        assert self.max_size != 0 # stop the program if no data
        
        self.is_train = is_train

        self.local_search = local_search # disabled
        
        # it is suggested to use numpy.array or pandas.DataFrame
        # to avoid out of memory error
        super(ProblemDataset, self).__init__()

    def __getitem__(self, idx):
        #pid = 0 # program of structure pid
        # pid = torch.randint(0,8,(1,) ).item() # 6 problems now, add special problems? TODO
        p,info,solu_len = self.get_program(idx)
        return p,info,solu_len
    
    def get_program(self,pid,parameter = [1.0]):
        # coefficient = torch.FloatTensor(3).uniform_(-2, 2) # a b c
        #coefficient = torch.FloatTensor(1).uniform_(1, 2) # a b c
        
        #a=coefficient[0].item() #TODO 0.24593424797058105 too long?
        
        # coefficient = torch.FloatTensor(5).uniform_(0.95, 1.05) # a b c
        # a1=coefficient[0].item()
        # a2=coefficient[1].item()
        # a3=coefficient[2].item()
        # a4=coefficient[3].item()
        # a5=coefficient[4].item()
        constraint = None# if is_train and local_search, constraint is active
        #parameter = [a1]#
        # parameter = [1.0] # test searching ability
        bit_base = 32
        if pid==0: # problem 1   OK
            solu_bits = [bit_base,bit_base] # x0, x1
            special = 'p1'
            obj_func = obj1
            g = g1
            h = h1
            if self.is_train: # in training and validating
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0]
                func_star = obj1_optimal
        elif pid==1: # problem 2
            solu_bits = [bit_base for i in range(5)]
            special = 'p2'
            obj_func = obj2
            g = g2
            h = h2
            if self.is_train:
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0]
                func_star = obj2_optimal
        elif pid==2:# problem 3
            solu_bits = [bit_base for i in range(6)] # x0, x1
            special = 'p3'
            obj_func = obj3
            g = g3
            h = h3
            if self.is_train: # in training and validating
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0]# in testing, the parameter should be fixed
                func_star = obj3_optimal
        elif pid==3:# problem 4
            solu_bits = [bit_base for i in range(5)] # TODO, 20 is dangerous
            special = 'p4'
            obj_func = obj4
            g = g4
            h = h4
            if self.is_train: # in training and validating
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0]# in testing, the parameter should be fixed
                func_star = obj4_optimal
        elif pid==4:# problem 5
            # solu_bits = [32,32,1,1,1] #
            solu_bits = [bit_base,bit_base,1,1,1] #
            special = 'p5'
            obj_func = obj5
            g = g5
            h = h5
            if self.is_train: # in training and validating
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0] # in testing, the parameter should be fixed
                func_star = obj5_optimal
        elif pid==5:# problem 6
            #parameter = [a1]#
            solu_bits = [1 for i in range(10)] #
            special = 'p6'
            obj_func = obj6
            g = g6
            h = h6
            if self.is_train: # in training and validating
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0] # in testing, the parameter should be fixed
                func_star = obj6_optimal
        elif pid==6:# problem 7
            solu_bits = [bit_base] #
            special = 'p7'
            obj_func = obj7
            g = g7
            h = h7
            if self.is_train: # in training and validating
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0] # in testing, the parameter should be fixed
                func_star = obj7_optimal
        elif pid==7:# problem 8
            solu_bits = [bit_base for i in range(100)] #
            special = 'p8'
            obj_func = obj8
            g = g8
            h = h8
            if self.is_train: # in training and validating
                func_star = 'None'
                if self.local_search:
                    constraint = cons_basic
            else:# in testing
                # parameter = [1.0]# in testing, the parameter should be fixed
                func_star = obj8_optimal
        # print (pfunc(0,parameter) )
        para_str = "parameter:\n"+str(parameter)
        
        func_str = inspect.getsource(obj_func)
        g_str = inspect.getsource(g)
        h_str = inspect.getsource(h)
        p = para_str+'\n'+func_str+g_str+h_str+'<|endofprompt|>'# TODO parameter end?
        
        if self.is_train: # in training and validating
            func_star_str = 'None'
        else:# in testing
            func_star_str = inspect.getsource(func_star)
        # print (1111)
        if self.is_train and self.local_search:
            constraint_str = inspect.getsource(constraint)
        else:
            constraint_str = 'None'
        
        info = {"solu_bits": solu_bits,
                "f": func_str,
                "g": g_str,
                "h": h_str,
                "func_star": func_star_str,# this is not used in training, only necessary in testing
                "parameter": parameter,
                "cons": constraint_str,
                "special": special}
        info = json.dumps(info)
        solu_len = sum(solu_bits)
        
        return p, info, solu_len
    
    def __len__(self):
        return self.max_size
    
    def getItem(self, pid):
        p,info,solu_len = self.get_program(pid)
        return p,info,solu_len

def collate_fn(batch,device,pad_token,enc):
    #enc.encode(txt, allowed_special="all")
    srcs = []
    infos = []
    max_solu_len = None
    solu_lens = []
    for src,info,solu_len in batch:
        # input_data = input_data.transpose(0,1).to(device)
        en_src = torch.tensor(enc.encode(src, allowed_special="all"), device=device)
        srcs.append(en_src)
        infos.append(info)
        if max_solu_len is None:
            max_solu_len = solu_len
        else:
            max_solu_len = solu_len if solu_len>max_solu_len else max_solu_len
        solu_lens.append(solu_len)
        # print (target)
        # raise
        # input_sizes[i] = seq_length
        # input_percentages[i] = seq_length #/ float(max_seq_len)
        # target_sizes[i] = len(target)
        # print (len(target))
        #targets.append(torch.tensor(ast.literal_eval( str(target) ),dtype=torch.int64,device=device))
    # The feature is in [-1,1] due to normalization, I pad input using 2
    #print (srcs)
    # srcs.append(torch.tensor(enc.encode("666", allowed_special="all"), device=device))
    srcs = pad_sequence(srcs, padding_value = pad_token).transpose(0,1)
    # print (solu_lens)
    # print (torch.ones(bsz,1))
    remain_bits = torch.tensor(solu_lens,device=device,dtype=torch.int32).view(-1,1)# no need to specify dtype, when moved to cuda it becomes torch.int32
    
    return srcs,infos,max_solu_len,remain_bits # , targets

class ProblemDataLoader(DataLoader):
    def __init__(self, train_iter_map, 
                 batch_size, 
                 device, 
                 sampler=None, 
                 shuffle=None, 
                 #pin_memory=False, 
                 pad_token=2,
                 enc=None):
        super(ProblemDataLoader, self).__init__(train_iter_map,
        batch_size = batch_size,
        shuffle = shuffle,
        sampler = sampler,
        collate_fn = lambda batch: collate_fn(batch,device,pad_token,enc),
        )

class ProblemDatasetOffline(Dataset):
    def __init__(self,
                 env_data,
                 mode="train",
                 data_dir = 'data/'
                 ):
        # read all files names into array
        # sample files
        # import os
        
        self.data_array = None
        self.env_data = env_data

        self.df = None
        for pid in range(8):
            if mode=="train":
                csv_filename = data_dir+f'train_data_problem{pid}.csv'
            elif mode=="valid":
                csv_filename = data_dir+f'valid_data_problem{pid}.csv'
            else:
                raise
            df = pd.read_csv(csv_filename)
            if self.df is not None:
                self.df = pd.concat([self.df, df], axis=0, ignore_index=True)
            else:
                self.df = df
        super(ProblemDatasetOffline, self).__init__()

    def __getitem__(self, idx):
        # p = self.data_array[idx][0] # string
        # info = self.data_array[idx][1] # string
        # solu_len = self.data_array[idx][2] # string
        # x_ref = self.data_array[idx][3] # string
        pid = self.df.iloc[idx]['pid'] #columns = ['pid', 'parameter', 'optimal_x']
        parameter = eval( self.df.iloc[idx]['parameter'] )
        x_ref = self.df.iloc[idx]['optimal_x'] #eval( self.df.iloc[idx]['optimal_x'] )
        # print (parameter,type(parameter))
        p,info,solu_len = self.env_data.get_program(pid,parameter)
        solu_len = str(solu_len) # string
        # print (type(solu_len))
        # print (solu_len)
        return p,info,solu_len,x_ref
    
    def __len__(self):
        # return len(self.data_array)
        return len(self.df)

def collate_fn_offline(batch,device,src_pad,pad_token,enc,sos_token,eos_token):
    srcs = []
    tgt = []
    for src,info,solu_len_str,ref in batch:
        srcs.append(torch.tensor(enc.encode(src, allowed_special="all"),  device=device) )
        x_tokens = enc.encode(ref, allowed_special="all")
        tgt.append(torch.tensor([sos_token] + x_tokens+[eos_token], device=device) )
        
    # srcs.append(torch.tensor(enc.encode("666", allowed_special="all"), device=device))
    srcs = pad_sequence(srcs, padding_value = src_pad).transpose(0,1)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt, padding_value = pad_token).transpose(0,1)
    return srcs,tgt

class ProblemDataLoaderOffline(DataLoader):
    def __init__(self, train_iter_map, 
                 batch_size, 
                 device, 
                 sampler=None, 
                 shuffle=None,
                 src_pad = None,
                 pad_token=None,
                 enc=None,
                 sos_token=None,
                 eos_token = None):
        super(ProblemDataLoaderOffline, self).__init__(train_iter_map,
        batch_size = batch_size,
        shuffle = shuffle,
        sampler = sampler,
        collate_fn = lambda batch: collate_fn_offline(batch,device,
                                                      src_pad,pad_token,
                                                      enc,sos_token,eos_token),
        )
