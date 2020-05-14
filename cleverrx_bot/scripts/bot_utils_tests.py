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
with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
    raw_data = pickle.load(file)

short_raw_data_list = [raw_data[i] for i in list(raw_data.keys())[0:6]]
short_raw_data_dict = {i:raw_data[i] for i in list(raw_data.keys())[0:6]}
example = short_raw_data_dict[list(short_raw_data_dict.keys())[0]]
list(example.keys())

short_raw_data[69] = {'tweet': 'I love dicks', 'topic_links': [], }
short_raw_data_df = pd.DataFrame.from_dict(short_raw_data_dict, orient = 'index')


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#testing __init__
list_preprocessor = butils.Comment_data_preprocessor(short_raw_data_list, 'tweet', tokenizer)
dict_preprocessor = butils.Comment_data_preprocessor(short_raw_data_dict, 'tweet', tokenizer)
df_preprocessor = butils.Comment_data_preprocessor(short_raw_data_df, 'tweet', tokenizer)

list_preprocessor.input_df
dict_preprocessor.input_df
df_preprocessor.input_df

#testing tokenizing datasets
keyword_dataset = dict_preprocessor.prepare_keyword_dataset(dict_preprocessor.input_df, 'id', 'text', 'topic_links', key = 'type_no_sentiment_cluster_keywords', cluster = True)
keyword_dataset
dict_preprocessor.prepared_datasets['type_no_sentiment_cluster_keywords']

#testing __getitem__
sample = dict_preprocessor[0] #should return no active dataset because it has not been set yet
dict_preprocessor.set_active_dataset('type_no_sentiment_cluster_keywords')
sample = dict_preprocessor[0]
data_sample = dict_preprocessor.active_dataset.loc[0]

text, keywords = dict_preprocessor[0]
dict_preprocessor.tokenizer.decode(keywords)
dict_preprocessor.tokenizer.decode(text)

dict_preprocessor.set_get_type('prepend_space')
text = dict_preprocessor[0]
dict_preprocessor.tokenizer.decode(text)

dict_preprocessor.set_get_type('prepend_nospace')
text = dict_preprocessor[0]
dict_preprocessor.tokenizer.decode(text)

###loading data
dict_preprocessor.set_get_type('keyword')
keyword_loader = DataLoader(dict_preprocessor, batch_size = 1, collate_fn = dict_preprocessor.collate_fn)
batches = list(keyword_loader)
bacth = batches[6]
texts, keywords = batch

keywords

dict_preprocessor.set_get_type('prepend_space')
prepend_loader = DataLoader(dict_preprocessor, batch_size = 2, collate_fn = dict_preprocessor.collate_fn)
batch = next(iter(prepend_loader)



#%%  test forward pass
inputs, labels = (batch, batch[0])
torch.cuda.is_available()
test = models.GPT2Model_bagofctrl.from_pretrained("gpt2")
device = "cpu"
test.wpe
len(test(batch, device))
outputs = test(batch)[0].shape #logits
test(batch)[1].shape #hidden states
labels = batch[0].shape
test(batch, labels = batch[0])[0]

#%% generation work
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


logits = test((bos_tokens, keyword_tokens), device)[0][:, -1, :]

torch.topk(logits, 20)[0][:, -1, None]



i
