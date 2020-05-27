import bot_utils as butils
import bot_models as models
import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer
import torch.nn.functional as F
import json
#TODO
# - fix generation with no stem
# - add temperature to generation

#%%
model_path = '../saved_models/'
model_name = 'batch_051220_keyword_types_sentiment_nocluster'

model = models.GPT2Model_bagofctrl.load(model_path + model_name)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#%% generation on fly
prompt1 =
prompt2 =
prompt3 =

output1 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt1, 50, top_k = 5, top_p = 0, num_return_sequences = 1)
output2 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt2, 50, top_k = 5, top_p = 0, num_return_sequences = 1)
output3 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt3, 50, top_k = 5, top_p = 0, num_return_sequences = 1)

output1 =
output2 =
output3 =

#%% generation test
test_json1 = json.load(open('../data/facebookgroups.json', 'rb'))
test_json2 = json.load(open('../data/facebookpages.json', 'rb'))

test_data1 = pd.DataFrame(test_json1)
test_data2 = pd.DataFrame(test_json2)

test_data1
test_data2
#sample outputs
# ['insurance plans for #healthcare companies will be on the rise for many years. this is why we need a #healthtech system for all. https://t.co/fkzvjkqzfk https://t.co/',
#  'insurance companies can use GoodRx as a product, but they are not a pharmacy.  they are a pharmacy.  the cost of a pharmacy is more than a pharmacy means.  the price of a pharmacy is more than a prescription.  the',
#  'insurance for the holidays is not a good choice. https://t.co/qkzfkqzgvq!!!!!!!!!!!!!!!!!!!!!!!!!',
#  "insurance costs have been a problem since 2020. it's time to fight the #healthcare industry. #healthcare #hcldr https://t.co/qhxwzqxzfk!!!!!!!",
#  'insurance costs for patients are over $10,000, but a new blog by @sabcsm https://t.co/xwvzgwjzgj via @youtube!!!!!!!!!!!']
