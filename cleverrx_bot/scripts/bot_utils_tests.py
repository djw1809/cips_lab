import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bot_utils as butils
import pickle
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset




# %%
#Data_handler tests

#testing __init__
test_str = 'This is a test string. I need to make it longer. So Im typing bullshit. Maybe this is long enough.  We will see'

test_json = [{'id': 1, 'tweet_text': 'This is the first tweet', 'keywords':'first'},
            {'id': 2, 'tweet_text': 'this is the second tweet', 'keywords':'second'},
            {'id': 3, 'tweet_text': np.nan, 'keywords':'third'},
            {'id': 4, 'tweet_text': 'this is the fourth tweet', 'keywords': [], 'extra_argument': 'a'}]


test_df = pd.DataFrame(test_json)


json_dataset = butils.Comment_data_preprocessor(test_json, 'id', 'tweet_text', 'keywords')
json_dataset.input_df

df_dataset = butils.Comment_data_preprocessor(test_df, 'id', 'tweet_text', 'keywords')
df_dataset.input_df

#%%
#testing df_to_corpus
json_dataset.corpus
df_dataset.corpus


json_dataset.df_to_corpus()
df_dataset.df_to_corpus()

json_dataset.corpus
df_dataset.corpus

#%%
#testing input_df to tokenized_df
synonym_df = pd.read_csv('../../data/new_topics.csv')
synonym_dict = synonym_df.set_index('0').T.to_dict('records')[0]

full_data = pd.read_csv('tweets_topics.csv')
small_data = full_data.loc[0:100, :]
small_data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

small_data_preprocessor_synonym = butils.Comment_data_preprocessor(small_data, 'id', 'text', tokenizer, keyword_field ='topics', synonym_dict = synonym_dict)
#check if keyword lists are really lists and not strings of lists
small_data_preprocessor_synonym.df_to_tokenized_df(number_of_keywords = 3)
small_data_preprocessor_synonym.tokenized_df.loc[0:4, :]


small_data_preprocessor_nosynonym = butils.Comment_data_preprocessor(small_data, 'id', 'text', tokenizer, keyword_field ='topics')
small_data_preprocessor_nosynonym.df_to_tokenized_df(number_of_keywords = 3)
small_data_preprocessor_nosynonym.tokenized_df.loc[0:4, :]

##testing dataset class/loader
df = small_data_preprocessor_synonym.tokenized_df



dataset = butils.Comment_dataset(df, 'prepended_token_ids')
loader = DataLoader(dataset, batch_size = 2, collate_fn = small_data_preprocessor_synonym.collate)

dataset[5]

for i in loader:
    print(i)
