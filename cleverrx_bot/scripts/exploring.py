# %%
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import bot_utils as butils
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss


#%%
blah = [1,2,3,4,5]
blah[:-1]
blah[1:]
#%%
prompt = "Donald Trump is the"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.all_special_tokens
tokenizer.max_len
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.encode(prompt, return_tensors="pt")

output = butils.generate_(model, tokenizer, prompt, 50, num_beams = 1, temperature = None, top_k = None, top_p = .9, repetition_penalty = 10, num_return_sequences = 5, print_ = True, stop_token = '.')


#%%
int = 1
insert_string = 'dicks'

print("I have {} {}.".format(int, insert_string))
#%%
text = "a bunch of text with a period here. I dont want any of this"
text = text[: (text.find('.') + 1)]
text
#%%
blah = torch.tensor([[1,2,3]])
blah2 = torch.tensor([[1],[2],[3]])
blah
blah2
blah.shape
blah2.size()
blah.squeeze()
blah.squeeze().size()
blah2.squeeze()
blah2.squeeze().size()

test = [1,2,3

#%%
better_data = pd.read_csv('../data/tweets_topics.csv')
better_data



#%%
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
type(model.config)

text = "lazy people have trouble with"


#%%
#synonym_dict =
training_set_path = '../data/tweets_topics.csv'
data = pd.read_csv(training_set_path)
data = data.loc[0:100, :]
pre_processor = butils.Comment_data_preprocessor(data, 'id', 'text', tokenizer, 'topics')
tokenized_comments = pre_processor.df_to_tokenized_df(number_of_keywords = 1)
dataset = butils.Comment_dataset(tokenized_comments, 'prepended_token_ids')







def collate(batch):
    if tokenizer._pad_token is None:
        return pad_sequence(batch, batch_first=True)
    return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)


training_loader = DataLoader(dataset, shuffle = True, num_workers = 1, batch_size = 2, collate_fn = collate)



test_batch = next(iter(training_loader))
test_batch
tokenizer.decode(test_batch[1])


inputs, labels = (test_batch, test_batch)
output = model(inputs, labels = labels)
len(output)
output[0].shape # loss
loss = output[0]
output[1].shape # lm_logits
lm_logits = output[1]
len(output[2])
output[2][1].shape # past

shifted_logits = lm_logits[:, :-1, :] #strip off last element in eahc output distribution
shifted_labels = labels[:, 1:] #strip off first element in rach label
#shifting in above manner means that



test_logits = torch.tensor([ [[1,2,1,1],[2,2,2,2],[3,3,3,3]], [[4,4,4,4], [5,5,5,5], [6,6,6,6]]])
test_logits.shape

test_logits[0,:-1, :].contiguous()



blah = torch.tensor([1,2,3,4,5])
blah[:-1]

w = torch.tensor([[1,1,1,1],[2,2,2,2]])
nd = w.size(-2)
ns = w.size(-1)
bias = torch.tril(torch.ones((5, 5), dtype=torch.uint8)).view(1, 1, 5, 5)
mask = bias[:, :, ns-nd : ns, :ns]
mask
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
