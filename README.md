# Transformer for Mathematical Programming
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


This code is for an experiment about using transformer as universal optimizer for mathematical programs.

## Environment

In the experiment, I used python 3.9 and cuda-version pytorch 2.4.1.

I used tiktoken 0.9.0 as the tokenizer.

The code is tested on a single-machine with 4 V100 GPUs and Ubuntu system.

## Data
I have already put the training and validation data in the ./data dir. 

If you would like to generate training data, please install gurobipy to make ./generateData.py work. 
Run generateData.py to generate data, and move the data to ./data dir if you'd like to generate your own data.

## Training and validation
Run ./main_func_SL.py with train_model() to train the transformer. 

The best validated model will be saved at ./models/model_prod.pt.

The training and validation losses are saved at ./log/model_prod.log.

## Testing
Run ./main_func_SL.py with test_model() to test the transformer on the 8 problems. 
It will load the model at ./models/model_prod.pt.

The inference (testing) results are saved at ./log/model_prod_test.log.

## Note
There is a note about the experiment:
[Transformer for Mathematical Programming](./note/transopt2025.pdf).

To cite the note:

@misc{jiasen2025transopt,
      title={Transformer for Mathematical Programming}, 
      author={Jiasen Wang},
      year={2025},
      url={./note/transopt2025.pdf}, 
}
