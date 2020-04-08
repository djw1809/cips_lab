# %%
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import bot_utils as butils
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset


# %%
a = torch.tensor([1,2,3])
print(a)

#%%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
text = "lazy people have trouble with"


tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)
token_ids
tokenizer.decode(token_ids)
#%%
###one predeiction

tokens = tokenizer.encode(text)
token_tensor = torch.tensor([tokens])
model = model.eval()
output = model(token_tensor)
output[0].shape
len(output[1])
predictions = model(token_tensor)[0]
predictions.shape
next_word_probabilities = predictions[:, 5, :]
next_word_probabilities.shape
token_index = torch.argmax(next_word_probabilities)
token_index
tokenizer.decode([token_index])


#%%
##looped prediction without past
input_tokens = tokenizer.encode(text)
input_tokens
generated = input_tokens
for i in range(30):
    input = torch.tensor(generated).unsqueeze(0)
    predictions = model(input)[0]
    predictions.shape
    next_word_probabilites = predictions[:, -1, :]
    token_index = torch.argmax(next_word_probabilites)
    token_index
    tokenizer.decode([token_index])
    generated += [token_index.tolist()]
    generated

print(tokenizer.decode(generated))




#%%
##looped prediction with past
generated_ = tokenizer.encode(text)
context = torch.tensor([generated_])
past = None

for i in range(30):
    output, past = model(context, past=past)
    token = torch.argmax(output[..., -1, :])

    generated_ += [token.tolist()]
    context = token.unsqueeze(0)

    sequence = tokenizer.decode(generated)

print(sequence)


#%% Our Data
raw_data = pd.read_csv('../../data/new_topics.csv')
raw_data
with open('../../data/topics_index.pkl', 'rb') as file:
    topics_index = pickle.load(file)

better_data = pd.DataFrame(columns = ['id', 'text', 'topics'])
for i in range(len(list(topics_index.keys()))):
    key = list(topics_index.keys())[i]
    tweet = topics_index[key]
    text = tweet['tweet']
    id = key
    topics = list(set().union(tweet['cost_list'],
                            tweet['card_list'],
                            tweet['customers_list'],
                            tweet['health_list'],
                            tweet['inhaler_list'],
                            tweet['insurance_list'],
                            tweet['medication_list'],
                            tweet['patients_list'],
                            tweet['religion_list'],
                            tweet['segment_list'],
                            tweet['service_list'],
                            tweet['transparency_list'],
                            ))
    row = {'id': id, 'text':text, 'topics':topics}
    better_data.loc[i, :] = row

better_data.to_csv('tweets_topics.csv')

dataset = butils.Data_handler(better_data)

print(dataset.raw_text)
dataset.raw_df

#%%%
test_tweet = better_data.loc[0]


#%%
