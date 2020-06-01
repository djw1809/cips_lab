import torch
import pickle
import json
import pandas as pd
import numpy as numpy
import transformers
from transformers import GPT2Tokenizer
import bot_utils as butils
import bot_models as models
#%%

class FacebookGenerator():
    '''class to easily generate on fly/pass generations to other applications'''
    def __init__(self, model_path, model_class, keyword_list, tokenizer):
        '''model_path (str): path to folder where model is stored
           model_class (cls): the class of the model instance to be used.  Needs a load method that can accept a folder path and find the model weights in that folder.  Should also have a generate method.
           keyword_list (iterable): a list of keywords that the model was trained on'''
        self.model = model_class.load(model_path)
        self.keywords = keyword_list
        self.tokenizer = tokenizer

    def process_incoming_comment(self, comment):
        '''needs to take as input a comment and return as output the keywords in the keyword list that are in the comment'''

    def generate(prompt, max_length, top_k = None, top_p = None, num_return_sequences = 5, min_keep = 1, filter_value = -float("Inf")):
        output = self.model.generate(self.tokenizer, prompt, max_length, top_k, top_p, num_return_sequences, min_keep, filter_value)
        return output

    def process_list_of_comments(self, comment_list):
        output = [self.process_incoming_comment(comment) for comment in comment_list]
        return output


#%%
model_path = '../saved_models/'
model_name1 = 'batch_051220_keyword_types_sentiment_cluster'
model_name2 = 'batch_051220_keyword_types_nosentiment_cluster'
model_name3 = 'batch_051220_keyword_types_sentiment_nocluster'

model1 = models.GPT2Model_bagofctrl.load(model_path + model_name1)
model2 = models.GPT2Model_bagofctrl.load(model_path + model_name2)
model3 = models.GPT2Model_bagofctrl.load(model_path + model_name3)


facebook_groups = pickle.load(open('../data/facebook_groups/topics_index_bots_fbgroups.pkl','rb'))
facebook_pages = pickle.load(open('../data/facebook_pages/topics_index_bots_fbpages.pkl', 'rb'))

prompt1 = 
prompt2 =
prompt3 =
