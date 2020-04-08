import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import bot_utils as butils
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
synonym_dict =
data = pd.read_csv('../../data/tweets_topics.csv')
pre_processor = butils.Comment_data_preprocessor(data, 'id', 'text', tokenizer, 'topics')
tokenized_comments = pre_processor.df_to_tokenized_df(number_of_keywords = 1)
dataset = Comment_datsaset(tokenized_comments, 'token_ids')

parameter_dict = {}


parameter_dict['training_set'] = tokenized_comments
parameter_dict['epochs'] = 1
parameter_dict['num_worker'] = 1
parameter_dict['batch_size'] =
parameter_dict['learning rate'] =
parameter_dict['weight_decay'] =
parameter_dict['eps'] =
parameter_dict['warmup_steps'] =

model = GPT2LMHeadModel.from_pretrained()

output = butils.train(dataset, tokenizer,
                               parameter_dict['epochs'],
                               parameter_dict['num_worker'],
                               parameter_dict['batch_size'],
                               parameter_dict['learning_rate'],
                               parameter_dict['weight_decay'],
                               parameter_dict['eps'],
                               parameter_dict['warmup_steps'],
                               model)
