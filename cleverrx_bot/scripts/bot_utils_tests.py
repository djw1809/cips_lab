import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bot_utils as butils
import bot_models as models
import pickle
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset,DataLoader



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
#synonym_df = pd.read_csv('../data/new_topics.csv')
#synonym_dict = synonym_df.set_index('0').T.to_dict('records')[0]


full_data = pd.read_csv('../data/tweets_topics.csv')
len(full_data)
full_data
small_data = full_data.loc[0:100, :]
small_data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

small_data_preprocessor = butils.Comment_data_preprocessor(small_data, 'id', 'text', tokenizer, keyword_field ='topics')
#check if keyword lists are really lists and not strings of lists
small_data_preprocessor.df_to_tokenized_df(number_of_keywords = None)

dataset = butils.bag_words_ctrl_Dataset(small_data_preprocessor)

###loading data
loader = DataLoader(dataset, batch_size = 2, collate_fn = dataset.collate)

dummy_embedding = torch.nn.Embedding(dataset.tokenizer.vocab_size, 10)

batch = next(iter(loader))

inputs, labels = (batch, batch[0])
torch.cuda.is_available()

#%%  test forward pass
test = models.GPT2Model_bagofctrl.from_pretrained("gpt2")
test.wpe
len(test(batch))
outputs = test(batch)[0].shape #logits
test(batch)[1].shape #hidden states
labels = batch[0].shape
test(batch, labels = batch[0])[0]

#%%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
prompt = (['diabetes', 'insurance'], 'diabetes is')
keyword_tokens = []
keywords, bos = prompt
for keyword in keywords:
    keyword_tokens = keyword_tokens + tokenizer.encode(keyword)

keyword_tokens
keyword_tokens = [torch.tensor(keyword_tokens)]
bos_tokens = torch.tensor(tokenizer.encode(bos)).unsqueeze(0)
bos_tokens


test((bos_tokens, keyword_tokens), device)[0][:, -1, :].size()

batch = (bos_tokens, [keyword_tokens])

i
