# import regex as re
import unicodedata
import tiktoken

def get_enc():
    '''
    Add special tokens.
    Adapted from https://github.com/openai/tiktoken
    '''
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    # raise
    # In production, load the arguments directly instead of accessing private attributes
    # See openai_public.py for examples of arguments for specific encodings
    # print ({**cl100k_base._special_tokens})
    # {'<|endoftext|>': 100257, 
    #  '<|fim_prefix|>': 100258, 
    #  '<|fim_middle|>': 100259, 
    #  '<|fim_suffix|>': 100260, 
    #  '<|endofprompt|>': 100276}
    enc = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|num_start|>": 100264, # start of a number
            "<|num_end|>": 100265, # end of a number
            "<|start|>": 100266, # start of a solution, not used
            "<|end|>": 100267,# end of a solution, not used
            "<|blank|>": 100268,# source pad
        }
    )
    return enc
    
# def is_fullwidth(string):
#     for char in string:
#         if 'FULLWIDTH' in unicodedata.name(char):
#             return True
#     return False
