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
#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model_path = '../saved_models/'
#
# model_name1 = 'batch_051220_keyword_types_sentiment_cluster'
# model_name2 = 'batch_051220_keyword_types_nosentiment_cluster'
model_name3 = 'batch_051220_keyword_types_sentiment_nocluster'



#save_name_1 = 'test'
# save_name_2 = '_081120_insurance'
save_name_3 ='_040621_clever'


#model1 = models.GPT2Model_bagofctrl.load(model_path + model_name1)
# model2 = models.GPT2Model_bagofctrl.load(model_path + model_name2)
model3 = models.GPT2Model_bagofctrl.load(model_path + model_name3)

model_dict = {model_name3 + save_name_3:model3} #model_name2+save_name_2:model2, model_name3+save_name_3:model3}

k_list = [100,120,140,160,180,200]
p_list = [.3, .4, .5, .6, .7]
length = 50
num_return_sequences = 30
temperature = 1.3
repetition_penalty = 2.5
prompts = [(['insurance-'], 'insurance is'),
           (['card+'], 'i use'),
           (['card+'], 'im trying'),
           (['insurance-'], 'pharma companies'),
           (['insurance-'], 'prescription costs'),
            (['insurance-'], 'the drug industry')]


for key in model_dict.keys():
    output = butils.parameter_sweep(model_dict[key], tokenizer, length, k_list, p_list, prompts, key, num_return_sequences, temperature = temperature, repetition_penalty = repetition_penalty)
