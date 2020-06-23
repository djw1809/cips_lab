import torch
import pickle
import json
import pandas as pd
import numpy as numpy
import transformers
from transformers import GPT2Tokenizer
from topic_link_creation import TopicLinkCreation
import bot_utils as butils
import bot_models as models

model_name1 = 'batch_051220_keyword_types_sentiment_cluster'
model_name2 = 'batch_051220_keyword_types_nosentiment_cluster'
model_name3 = 'batch_051220_keyword_types_sentiment_nocluster'

model1 = models.GPT2Model_bagofctrl.load(model_path + model_name1)
model2 = models.GPT2Model_bagofctrl.load(model_path + model_name2)
model3 = models.GPT2Model_bagofctrl.load(model_path + model_name3)

model_dict = {model_name1:model1, model_name2:model2, model_name3:model3}

k_list = [0,20,40,60,80,100,120,140,160,180,200]
p_list = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
length = 30
prompt1 = ['insurance-', 'insurance is']
prompt2 = ['card+', 'Use']


for key in model_dict.keys():
    output = butils.parameter_sweep(model_dict[key], length, k_list, p_list, prompt1, prompt2, key)
