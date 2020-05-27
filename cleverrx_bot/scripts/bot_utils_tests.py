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

#%% Data_handler tests
with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
    raw_data = pickle.load(file)


#%% regular data
raw_data = {1: {'tweet': 'a tweet'},
            2:{'tweet': 'a second tweet'},
            3:{'tweet': 'a third tweet'},
            4:{'tweet': 'a fouth tweet'},
            5:{'tweet': 'a fifth tweet'}}


#%% keyword data
short_raw_data_list = [raw_data[i] for i in list(raw_data.keys())[0:6]]
short_raw_data_list
short_raw_data_dict = {i:raw_data[i] for i in list(raw_data.keys())[0:6]}
short_raw_data_dict
short_raw_data_df = pd.DataFrame.from_dict(short_raw_data_dict, orient = 'index')
short_raw_data_df


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#%% testing __init__
list_preprocessor = butils.Comment_data_preprocessor(short_raw_data_list, 'tweet', tokenizer)
dict_preprocessor = butils.Comment_data_preprocessor(short_raw_data_dict, 'tweet', tokenizer)
df_preprocessor = butils.Comment_data_preprocessor(short_raw_data_df, 'tweet', tokenizer)
regular_dict_preprocessor = butils.Comment_data_preprocessor(raw_data, 'tweet', tokenizer)


regular_dict_preprocessor.input_df
list_preprocessor.input_df
dict_preprocessor.input_df
df_preprocessor.input_df

#%% testing tokenizing keyword datasets
keyword_dataset = dict_preprocessor.prepare_keyword_dataset(dict_preprocessor.input_df, 'id', 'text', 'topic_links', key = 'type_no_sentiment_cluster_keywords', cluster = True, sentiment = False)
keyword_dataset
(dict_preprocessor.prepared_datasets.keys())
dict_preprocessor.prepared_datasets['type_no_sentiment_cluster_keywords']

#%% testing tokenzing regular datasets
regular_dataset = regular_dict_preprocessor.df_to_tokenized_df(regular_dict_preprocessor.input_df)
regular_dataset

#%% loading regular_dataset into pytorch dataset object/testing getitem
regular_torch_dataset = butils.Comment_dataset(regular_dataset, 'token_ids', regular_dict_preprocessor.tokenizer)
regular_torch_dataset[2]
len(regular_torch_dataset)


#%% testing __getitem__
sample = dict_preprocessor[0] #should return no active dataset because it has not been set yet
dict_preprocessor.set_active_dataset('type_no_sentiment_cluster_keywords')
sample = dict_preprocessor[0]
sample
sample
data_sample = dict_preprocessor.active_dataset.loc[0]
data_sample

text, keywords = dict_preprocessor[0]
dict_preprocessor.tokenizer.decode(keywords)
dict_preprocessor.tokenizer.decode(text)

dict_preprocessor.set_get_type('prepend_space')
text = dict_preprocessor[0]
dict_preprocessor.tokenizer.decode(text)


dict_preprocessor.set_get_type('prepend_nospace')
text = dict_preprocessor[0]
dict_preprocessor.tokenizer.decode(text)

#%% loading keyword data
dict_preprocessor.set_get_type('keyword')
keyword_loader = DataLoader(dict_preprocessor, batch_size = 2, collate_fn = dict_preprocessor.collate_fn)
batches = list(keyword_loader)
batches
texts, keywords = batch
batch = batches[0]

keywords

dict_preprocessor.set_get_type('prepend_space')
prepend_loader = DataLoader(dict_preprocessor, batch_size = 2, collate_fn = dict_preprocessor.collate_fn)
batch = next(iter(prepend_loader)

#%% loading regular data
regular_loader = DataLoader(regular_torch_dataset, batch_size = 2, collate_fn = regular_torch_dataset.collate)
batches
batches[0]

#%%  test forward pass
inputs, labels = (batch, batch[0])
torch.cuda.is_available()
test = models.GPT2Model_bagofctrl.from_pretrained("gpt2")
device = "cpu"
outputs = test(inputs, device)
outputs[0].shape


#%% generation tests
model_path = '../saved_models/'
model_name = 'batch_051220_keyword_types_sentiment_nocluster'
model = models.GPT2Model_bagofctrl.load(model_path + model_name)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt1 = (['insurance-', 'card+'], 'insurance') #normal prompt
prompt2 = ([], 'insurance') #no keyword prompt
prompt3 = (['insurance-'],'') #no stem prompt

ouput11 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt1, 50, top_k = 5, top_p = 0, num_return_sequences = 1) #just top k
output12 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt1, 50, top_k = 0, top_p = .8, num_return_sequences = 1) #just nucleus
output13 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt1, 50, top_k = 5, top_p = .8, num_return_sequences = 1) #both

ouput21 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt2, 50, top_k = 5, top_p = 0, num_return_sequences = 1) #just top k
output22 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt2, 50, top_k = 0, top_p = .8, num_return_sequences = 1) #just nucleus
output23 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt2, 50, top_k = 5, top_p = .8, num_return_sequences = 1) #both

ouput31 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt3, 50, top_k = 5, top_p = 0, num_return_sequences = 1) #just top k
output32 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt3, 50, top_k = 0, top_p = .8, num_return_sequences = 1) #just nucleus
output33 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt3, 50, top_k = 5, top_p = .8, num_return_sequences = 1) #both






i
