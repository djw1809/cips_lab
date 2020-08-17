import pytest
import torch
import pickle
import data_processing
import bot_utils as butils
import pandas as pd
import json
from transformers import GPT2Tokenizer, BertTokenizer
from torch.utils.data import DataLoader, Dataset
##test for data processing

def test_count_hashtags():
    with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
        data = pickle.load(file)

    data = pd.DataFrame.from_dict(data, orient = 'index')
    data.index = range(len(data))
    test_data = data[0:5]
    output_dict = data_processing.count_hashtags(test_data, 'tweet')
    assert output_dict['#love.'] == 1


##test for Comment_data_preprocessor

##testing init
def test_list_preprocessor():

    with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
        raw_data = pickle.load(file)
    short_raw_data_list = [raw_data[i] for i in list(raw_data.keys())[0:6]]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    list_preprocessor = butils.Comment_data_preprocessor(short_raw_data_list, 'tweet', tokenizer, id_field = 'id')
    assert list_preprocessor.input_df.loc[0, 'service+_phrases'] == ['#love']

def test_dict_preprocessor():

    with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
        raw_data = pickle.load(file)
    short_raw_data_dict = {i:raw_data[i] for i in list(raw_data.keys())[0:6]}
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dict_preprocessor = butils.Comment_data_preprocessor(short_raw_data_dict, 'tweet', tokenizer, id_field = 'id')
    assert dict_preprocessor.input_df.loc[0, 'service+_phrases'] == ['#love']

def test_df_preprocessor():

    with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
        raw_data = pickle.load(file)

    short_raw_data_dict = {i:raw_data[i] for i in list(raw_data.keys())[0:6]}
    short_raw_data_df = pd.DataFrame.from_dict(short_raw_data_dict, orient = 'index')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    df_preprocessor = butils.Comment_data_preprocessor(short_raw_data_df, 'tweet', tokenizer)
    assert df_preprocessor.input_df.loc[0, 'service+_phrases'] == ['#love']


#testing tokenizing

def test_tokenize_keyword():

    with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
        raw_data = pickle.load(file)
    short_raw_data_dict = {i:raw_data[i] for i in list(raw_data.keys())[0:6]}
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dict_preprocessor = butils.Comment_data_preprocessor(short_raw_data_dict, 'tweet', tokenizer)
    dict_preprocessor.prepare_keyword_dataset(dict_preprocessor.input_df, 'id', 'text', 'topic_links', key = 'test', cluster = True, sentiment = False)
    assert set(dict_preprocessor.prepared_datasets['test'].loc[0, 'keyword_ids']) == set([15271, 9517])

def test_tokenize_regular():
    raw_data_regular = {1: {'tweet': 'a tweet'},
                2:{'tweet': 'a second tweet'},
                3:{'tweet': 'a third tweet'},
                4:{'tweet': 'a fouth tweet'},
                5:{'tweet': 'a fifth tweet'}}
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    regular_dict_preprocessor = butils.Comment_data_preprocessor(raw_data_regular, 'tweet', tokenizer)
    regular_dataset = regular_dict_preprocessor.df_to_tokenized_df(regular_dict_preprocessor.input_df)
    assert set(regular_dataset.loc[0, 'token_ids']) == set([64, 6126])

##tests for Comment_pair_dataset


##testing tokenizing
def test_tokenize_pair():
    with open('../data/pairs_v1.json', 'rb') as file:
        data = json.load(file)
    short_data = data[0:5]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = butils.Comment_pair_dataset(short_data, 'fb_post', 'tweet', tokenizer)
    assert dataset.active_data.loc[0, 'token_ids_sample1'][0] == 101


##testing loading into DataLoader
##generic load
def test_loading_pair():
    with open('../data/pairs_v1.json', 'rb') as file:
        data = json.load(file)
    short_data = data[0:5]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = butils.Comment_pair_dataset(short_data, 'fb_post', 'tweet', tokenizer)
    loader = DataLoader(dataset, collate_fn = dataset.collate, batch_size = 2, shuffle = True)
    batch = next(iter(loader))
    assert (batch[0][0].shape == batch[0][1].shape) and (batch[1][0].shape == batch[1][1].shape)

#asserting a max length
def test_max_len():
    with open('../data/pairs_v1.json', 'rb') as file:
        data = json.load(file)
    short_data = data[0:5]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = butils.Comment_pair_dataset(short_data, 'fb_post', 'tweet', tokenizer)
    dataset.max_len = 5
    loader = DataLoader(dataset, collate_fn = dataset.collate, batch_size = 1, shuffle = False)
    batch = next(iter(loader))
    assert (len(batch[0][0]) == len(batch[1][0]) == 5)


##testing changing get order
def test_pair_switch_order():
    with open('../data/pairs_v1.json', 'rb') as file:
        data = json.load(file)

    short_data = data[0:5]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = butils.Comment_pair_dataset(short_data, 'fb_post', 'tweet', tokenizer)
    loader = DataLoader(dataset, collate_fn = dataset.collate, batch_size = 2, shuffle = False)
    batch1 = next(iter(loader))
    dataset.set_get_type('sample2_first')
    loader = DataLoader(dataset, collate_fn = dataset.collate, batch_size = 2, shuffle = False)
    batch2 = next(iter(loader))
    assert (torch.all(torch.eq(batch1[0], batch2[1]))) and (torch.all(torch.eq(batch1[1], batch2[0])))
