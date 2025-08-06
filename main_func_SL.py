
import time
import numpy as np
np.seterr(all='ignore') # otherwise cos(nan) will give runtime error
import logging

import os
from contextlib import contextmanager

import torch
from torch.nn import functional as F
# from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.Label import get_enc
from utils.DataLoader import ProblemDataset, ProblemDataLoader, ProblemDatasetOffline, ProblemDataLoaderOffline
from utils.utils import make_model, run_epoch, reduce_tensor, run_model_test
from utils.LabelSmoothing import LabelSmoothing
from utils.SimpleLossCompute import LossCompute
from utils.Batch import Batch,BatchOffline #sampling and training dataset
from utils.Evaluation import MAE
import warnings
warnings.filterwarnings("ignore")

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
########################
# Parameters setting loop
########################
def getArgs(world_size):
    args = DotDict()
    ######train parameters
    args.save_every = 50 #
    args.num_epochs = 100
    args.batch_size = 16
    args.valid_size = 2000 #not used,
    args.test_size = 8 # 8 problems
    
    args.base_lr = 3e-4 #learning rate for Adam
    
    args.src_max_len = 4000 # default maximum source length, seems not used
    args.max_output_len = 900 # maximum output tokens
    
    args.pad_char = "<|blank|>" # pad
    args.sos_char = "<|start|>" # start of txt
    args.eos_char = "<|end|>" # end of txt

    args.enc = get_enc() # get tiktoken, including extra special tokens
    args.src_pad = args.enc._special_tokens["<|blank|>"] # for source
    args.pad_token = args.src_pad # for tgt, not should be consistent with enc
    args.sos_token = args.enc._special_tokens[args.sos_char]
    args.eos_token = args.enc._special_tokens[args.eos_char]

    args.num_layers = 6 #6 # Number of layers for debugging
    args.dim_emb = 64 #64   dim model embedding
    args.d_ff = 64 #64   dim feedforward
    args.num_heads = 2 #2 #8
    args.d_src = 100276 + 1 # 100276 is max number of tokens in enc
    args.dropout = 0.1 # dropout rate of transformer model

    args.debug = True # True means using less data for debugging, use False for real training
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0') #TODO multi-GPU
        args.cuda = True
        args.debug = False# has cuda, default production (not debug)
    else:
        args.device = torch.device('cpu')
        args.cuda = False
    
    args.last_path = "models/last_prod.pt"#storing last epoch model
    args.file_path = "models/model_prod.pt"#storing best model
    args.continue_from ="" #args.last_path #"" #args.last_path#"" #args.last_path #"" # args.file_path #"" last_path #args.file_path #""  #args.file_path #""
    
    args.name = 'model_prod.log' #for storing training log
    args.test_name = 'model_prod_test.log' #for storing testing log
    return args

###################
# Main training loop
###################
def train_model():
    world_size = torch.cuda.device_count()
    args = getArgs(world_size) # set args in this function
    if not os.path.exists("./log"):#TODO
        os.mkdir("./log")
    log_filename = "log/" + args.name
    logging.basicConfig(handlers=[logging.FileHandler(filename=log_filename, encoding='utf-8')],
                        format='%(message)s',
                        level=logging.INFO,
                        force=True)
    with open(log_filename, 'w'): # clear the log file if there is something
        pass
    logging.info("="*50)
    logging.info("THE EXPERIMENT LOG IS SAVED IN: " + "log/" + args.name)
    logging.info("="*50)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    logging.info(f"Number of GPUs detected: {world_size}")
    logging.info("Spawning training processes ...")
    mp.spawn(
        train_worker,
        args=(world_size,args),  # this is important, (ngpus,) is OK, (ngpus) is wrong
        nprocs=world_size,
    )

def train_worker(rank,world_size,args):
    log_filename = "log/" + args.name
    logging.basicConfig(handlers=[logging.FileHandler(filename=log_filename, encoding='utf-8')],
                        format='%(message)s',
                        level=logging.INFO,
                        force=True)
    if rank==0: torch.manual_seed(1) # If same seed, will sample the same on each cuda
    elif rank==1: torch.manual_seed(100)
    elif rank==2: torch.manual_seed(1000)
    elif rank==3: torch.manual_seed(10000)
    else: raise # means more than 4 GPUs, should modify the code to adjust to that
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    # dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=world_size)
    is_main_process = rank == 0
    with torch_distributed_zero_first(rank):
        env_data = ProblemDataset(is_train = True)
    
    tgt_vocab = args.d_src
    model = make_model(src_vocab = args.d_src,
                       tgt_vocab = tgt_vocab,
                       N = args.num_layers,
                       d_model = args.dim_emb,
                       d_ff = args.d_ff,
                       h = args.num_heads,
                       dropout = args.dropout)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # with torch.no_grad():
    if rank==0:logging.info(f"Number of parameters of the transformer: {num_parameters}")
    #Start from a specified checkpoint

    if rank==0:logging.info("Starts training from: " + args.continue_from + ". Space means from scratch.")
    model=model.to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    epoch_ckpt = 0
    if len(args.continue_from) > 0:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.continue_from, map_location=map_location)
        epoch_ckpt = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_train'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        del map_location
    with torch_distributed_zero_first(rank): data_train = ProblemDatasetOffline(env_data,mode="train",data_dir = 'data/')
    # offline_size = data_offline.__len__()# numbers of training sample stored offline
    # print (data_train.__len__() )
    train_dataloader = ProblemDataLoaderOffline(data_train,
                                batch_size=args.batch_size,
                                device=rank,
                                sampler= DistributedSampler(data_train),
                                shuffle=False,
                                src_pad=args.src_pad,
                                pad_token = args.pad_token,
                                enc=args.enc,
                                sos_token = args.sos_token,
                                eos_token=args.eos_token)
    with torch_distributed_zero_first(rank): data_valid = ProblemDatasetOffline(env_data,mode="valid",data_dir = 'data/')
    # print (data_valid.__len__() )
    valid_dataloader = ProblemDataLoaderOffline(data_valid,
                                batch_size=args.batch_size,
                                device=rank,
                                sampler= DistributedSampler(data_valid),
                                shuffle=False,
                                src_pad=args.src_pad,
                                pad_token = args.pad_token,
                                enc=args.enc,
                                sos_token = args.sos_token,
                                eos_token=args.eos_token)
    
    criterion = LabelSmoothing(
        size=tgt_vocab, padding_idx=args.pad_token, smoothing=0.1
    )
    criterion.cuda(rank) # original

    #with torch_distributed_zero_first(rank):train_data = ProblemDatasetTrain(None,args.train_size) # use None as placeholder
    
    min_valid_loss = np.inf
    for i in range(args.num_epochs):
        epoch = i+epoch_ckpt
        t0 = time.time()
        ##############train
        train_dataloader.sampler.set_epoch(epoch) # shuffle
        
        loss,tokens = run_epoch(
            (BatchOffline(b[0],b[1],args.src_pad,args.pad_token) for b in train_dataloader),
            model,
            LossCompute(model.module.generator,criterion),
            optimizer,
            mode="train")
        vector = torch.cat( [torch.tensor([loss],device=rank),torch.tensor([tokens],device=rank)] )
        train_loss = reduce_tensor(vector,dist)
        dist.barrier()
        if rank==0: logging.info(f"Epoch {epoch}, average training loss: {train_loss.item()} ")
        if args.cuda:
            torch.cuda.empty_cache()
        # if rank==0: logging.info(f"Epoch {epoch} Validation ====")
        ##############validation
        model.eval()
        loss,tokens = run_epoch(
            (BatchOffline(b[0],b[1],args.src_pad,args.pad_token) for b in valid_dataloader),
            model,
            LossCompute(model.module.generator,criterion),
            optimizer,
            mode="validation")
        vector = torch.cat( [torch.tensor([loss],device=rank),torch.tensor([tokens],device=rank)] )
        valid_loss = reduce_tensor(vector,dist)
        dist.barrier()
        if rank==0: logging.info(f"Epoch {epoch}, average validation gap: {valid_loss.item()} ")
        if args.cuda:
            torch.cuda.empty_cache()
        elapsed_time = time.time()-t0
        if rank==0: logging.info(f"Elapsed time for epoch {epoch} in seconds: {elapsed_time}")
        min_valid_loss > valid_loss.item()
        save = True if min_valid_loss > valid_loss.item() else False
        if save: # save best model
            min_valid_loss = valid_loss.item()
            if rank==0:# save best validated model
                #applies to DDP model, ref: distributed barrier checkpoints https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
                torch.save(model.state_dict(), args.file_path)
                logging.info(f"Epoch={epoch}, best model is updated and saved at: {args.file_path}")
        if epoch % args.save_every==0 and rank==0:
            torch.save({
            'epoch': epoch,
            'model_train': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, args.last_path)

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()

###################
# Main testing loop
###################
def test_model(last=False):
    world_size = torch.cuda.device_count()
    args = getArgs(world_size)
    if not os.path.exists("./log"):#TODO
        os.mkdir("./log")
    log_filename = "log/" + args.test_name
    logging.basicConfig(handlers=[logging.FileHandler(filename=log_filename, encoding='utf-8')],  # Specify encoding here
                                # logging.StreamHandler()  # Log to console as well,
                        # filemode='w+',
                        format='%(message)s',
                        level=logging.INFO,
                        force=True) # force=True is necessary, otherwise it only works for the FIRST logging basic config
    with open(log_filename, 'w'): # clear the log file if there is something
        pass
    
    tgt_vocab = args.d_src

    model = make_model(src_vocab = args.d_src,
                       tgt_vocab = tgt_vocab, #len(args.id2vocab),
                       N = args.num_layers,
                       d_model = args.dim_emb,
                       d_ff = args.d_ff,
                       h = args.num_heads,
                       dropout = args.dropout)
    rank = 0 # GPU 0
    model=model.to(rank)

    # for loading model with module in DDP, and run in single GPU. Test uses only one GPU.
    model = torch.nn.DataParallel(model) 
    
    if last:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.last_path, map_location=map_location)
    else:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.file_path, map_location=map_location)
        model.load_state_dict(checkpoint)
    model.eval()#for testing, write to function
    test_data = ProblemDataset(max_size=args.test_size,
                                src_max_len=args.src_max_len,
                                sos_token = args.sos_token,
                                eos_token = args.eos_token,
                                is_train = False)
    
    test_dataloader = ProblemDataLoader(test_data,
                                      batch_size=args.batch_size, # fixed, inference one by one
                                      device=rank,
                                      shuffle=False,
                                      pad_token=args.src_pad,
                                      enc=args.enc)
    
    pad_idx = args.pad_token
    eos_string = args.eos_char
    start_id = args.sos_token
    t0 = time.time()

    targets,preds = run_model_test((Batch(b[0], b[1], b[2],b[3], args.src_pad) for b in test_dataloader),
                                   model,
                                   pad_idx,
                                   eos_string,
                                   start_id,
                                   args.cuda,
                                   args.device,
                                   tgt_vocab,
                                   args)
    
    elapsed_time = time.time()-t0
    mae = MAE(targets,preds)
    # print (char_error_rate) # to log
    out_str = f"Mean absolute error in the inference is: {mae}"
    logging.info(out_str)
    logging.info(f"Elapsed time for inference testing in seconds: {elapsed_time}")

if __name__ == '__main__':
    # change archte parameter
    # train_model() # for training and storing models
    test_model() # for testing and storing test results in log