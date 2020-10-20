#%%
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import bot_utils as butils
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, EncoderDecoderModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
import bot_models as models
from topic_link_creation import TopicLinkCreation
import xlrd
import json
from random import sample
#%%
with open('../data/pairs_v3.pkl', 'rb') as file:
    data = pickle.load(file)

test_data = sample(data, 2000)
input_data = pd.DataFrame(test_data)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = butils.Comment_pair_dataset(test_data, 'tweet', 'fb_post', tokenizer)
dataset.max_len = 512
dataset.active_data

len(dataset[0][0])
len(dataset[0][1])



loader = DataLoader(dataset, batch_size = 2, collate_fn = dataset.collate)
next(iter(loader))


#%%
with open('../data/pairs_v2.pkl', 'rb') as file:
    data = pickle.load(file)
data = data[0:500]
sample1_field = 'fb_post'
sample2_field = 'tweet'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = butils.Comment_pair_dataset(data, sample1_field, sample2_field, tokenizer)
dataset.set_get_type('sample1_first')
dataset.max_len = tokenizer.max_len

dataset.raw_data[320]
#%%
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
model.config.to_dict()


#%%
with open('../data/pairs_v1.json', 'rb') as file:
    data = json.load(file)

short_data = data[0:5]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = butils.Comment_pair_dataset(short_data, 'fb_post', 'tweet', tokenizer)
dataset.max_len = 5

tokenizer.max_len




loader = DataLoader(dataset, collate_fn = dataset.collate, batch_size = 1, shuffle = True)
batch1 = next(iter(loader))
len(batch1[0][0])


batch1
if torch.all(torch.eq(batch1[0], batch2[1])) == True:
    print('dicks')
batch2[1] == batch1[0]



#%%
def blah(a,b):
    return a,b

def blah1(a,b):
    return (a,b)

blah(1,2) == blah1(1,2)
#%%
with open('../data/pairs_v1.json', 'rb') as file:
    data = json.load(file)

example = data[69]
example

df = pd.DataFrame(data)
df

#%%
with open('../data/facebookgroups.json', 'rb') as file:
    group_data = json.load(file)

with open('../data/facebookpages.json', 'rb') as file:
    page_data = json.load(file)

group_data[0].keys()
page_data[0].keys()
len(group_data)
len(page_data)

with open('../data/full_facebook_data.pkl', 'wb') as file:
    pickle.dump(group_data, file)



#%%
with open('../data/facebookgroups.json', 'rb') as file:
    data = json.load(file)

short_data = data[0:5]
short_data[0].keys()
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
preprocessor = butils.Comment_data_preprocessor(short_data, 'content', 'postid', tokenizer)
preprocessor.input_df
tokenized_df = preprocessor.df_to_tokenized_df(preprocessor.input_df)
tokenized_df





#%%
with open('../data/clusters.pkl', 'rb') as file:
    clusters = pickle.load(file)

clusters.keys()
#%%


#%%
raw_data_path = '../data/topics_index_bots_new_042820.pkl'
with open(raw_data_path, 'rb') as file:
    raw_data = pickle.load(file)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
preprocessor = Comment_data_preprocessor(raw_data, 'tweet', tokenizer)
preprocessor.input
#%%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')

with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
    raw_data = pickle.load(file)

short_raw_data_dict = {i:raw_data[i] for i in list(raw_data.keys())[0:6]}
dict_preprocessor = butils.Comment_data_preprocessor(short_raw_data_dict, 'tweet', tokenizer)
dict_preprocessor.set_active_dataset('type_no_sentiment_cluster_keywords')
keyword_dataset = dict_preprocessor.prepare_keyword_dataset(dict_preprocessor.input_df, 'id', 'text', 'topic_links', key = 'type_no_sentiment_cluster_keywords', cluster = True, sentiment = False)
dict_preprocessor.set_get_type('keyword')

example = dict_preprocessor[0]
example

#training
decoder_input_ids = torch.tensor(example[0]).unsqueeze(0) #the sequence
decoder_input_ids.size()
input_ids = torch.tensor(example[1]).unsqueeze(0) #keywords to the encoder
outputs = model(input_ids = input_ids, decoder_input_ids = decoder_input_ids, lm_labels = decoder_input_ids)

outputs[:2][0]


input_ids
#%%
def func(arg = 'a formatted string {}'.format('dicks')):
    print(arg)

func()


#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_ids = tokenizer.encode("Hello, I have a big dick.")
tokenizer.decode(input_ids)


#%%
some_strings = ['a string', 'another string', 'a third string with a really big dick']
a_file = open('test.txt', 'w')
for string in some_strings:
    a_file.write(string + '\n')
a_file.write('duh')
a_file.close()

#%%
clusters = pickle.load(open('../data/clusters.pkl', 'rb'))
clusters

tweet_classifier = TopicLinkCreation()
test_tweet = 'I have problems with my insurance'
output = tweet_classifier.build_graph(test_tweet)
output
#%%
model = models.GPT2Model_bagofctrl.from_pretrained('gpt2-medium')

#%%
fbgroup1 = pickle.load(open('../data/facebook_groups/topics_index_bots_fbgroups.pkl', 'rb'))
fbgroup2 = pickle.load(open('../data/only_topic_links/topics_link_bots_fbgroups.pkl', 'rb'))

fbgroup1[list(fbgroup1.keys())[0]]

fbgroup2[list(fbgroup2.keys())[0]]
#%%
facebook_groups = pd.read_csv('../data/facebook_data/facebookGroups.csv',  sep = ',')

facebook_pages = pd.read_csv('../data/facebook_data/facebookPages.csv')

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

for i in range(30s):
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

#%% inherting class methods

class A:

    @classmethod
    def blah(cls, string):
        return "This is {}".format(string)

class B(A):

    @classmethod
    def blah_(cls, string):
        output = super().blah(string)
        return output

class C():
    def __init__(self):
        self.list = [1,2,3]

    def function(self, default = len(self.list)):
        print(default + 1)

B.blah_('test')


#%%
with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
    new_data = pickle.load(file)

new_example = new_data[list(new_data.keys())[1]]
new_example
new_example.keys()

with open('../data/topics_index_old.pkl', 'rb') as file:
    old_data = pickle.load(file)

old_example = old_data[list(old_data.keys())[0]]
old_example.keys()


#%%
