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
model_name1 = 'batch_051220_keyword_types_sentiment_cluster'
model_name2 = 'batch_051220_keyword_types_nosentiment_cluster'
model_name3 = 'batch_051220_keyword_types_sentiment_nocluster'

model1 = models.GPT2Model_bagofctrl.load(model_path + model_name1)
model2 = models.GPT2Model_bagofctrl.load(model_path + model_name2)
model3 = models.GPT2Model_bagofctrl.load(model_path + model_name3)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#%% keyword tests

#%%sentiment no cluster
prompt2 = (['insurance-'],'insurance')
prompt3 = (['card+'], 'insurance')
prompt4 = (['insurance+'],'insurance' )
prompt5 = (['insurance-, card+'], 'insurance')

output2 = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt2, 50, top_k = 20, top_p = .7, num_return_sequences = 5)
output3 = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt3, 50, top_k = 20, top_p = .7, num_return_sequences = 5)
output4 = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt4, 50, top_k = 20, top_p = .7, num_return_sequences = 5)
output5 = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt5, 50, top_k = 20, top_p = .7, num_return_sequences = 5)

output2[0]
output3[0]
output4[0]
output5[0]
#%%nosentiment cluster
prompt6 = (['insurance', 'cancer'], 'cancer')
prompt7 = (['insurance', 'diabetes'], 'diabetes')
prompt8 = (['insurance', 'depression'], 'depression')

output6 = butils.generate_ctrl_bagofwords(model2, tokenizer, prompt6, 50, top_k = 20, top_p = .7, num_return_sequences = 5)
output7 = butils.generate_ctrl_bagofwords(model2, tokenizer, prompt7, 50, top_k = 20, top_p = .7, num_return_sequences = 5)
output8 = butils.generate_ctrl_bagofwords(model2, tokenizer, prompt8, 50, top_k = 20, top_p = .7, num_return_sequences = 5)

output6[0]
output7[0]
output8[0]
#%%sentiment cluster
prompt9 = (['insurance-', 'cancer'], 'cancer and insurance')
prompt10 = (['insurance-', 'diabetes'], 'diabetes and insurance')
prompt11 = (['insurance-', 'depression'], 'depression and insurance')

output9 = butils.generate_ctrl_bagofwords(model1, tokenizer, prompt9, 50, top_k = 20, top_p = .7, num_return_sequences = 5)
output10 = butils.generate_ctrl_bagofwords(model1, tokenizer, prompt10, 50, top_k = 20, top_p = .7, num_return_sequences = 5)
output11 = butils.generate_ctrl_bagofwords(model1, tokenizer, prompt11, 50, top_k = 20, top_p = .7, num_return_sequences = 5)

output9[0]
output10[0]
output11[0]

#%% keyword + stem tests
#sentiment no cluster


#nosentiment cluster



#sentiment cluster



output1 = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt1, 20, top_k = 20, top_p = .9, num_return_sequences = 5)
output2 = butils.generate_ctrl_bagofwords(model1, tokenizer, prompt2, 50, top_k = 5, top_p = 0, num_return_sequences = 5)
output3 = butils.generate_ctrl_bagofwords(model, tokenizer, prompt3, 50, top_k = 5, top_p = 0, num_return_sequences = 1)

output1[0]
output2 =
output3 =

#%% generation test
test_json1 = json.load(open('../data/facebook_groups/tfacebookgroups.json', 'rb'))
test_json2 = json.load(open('../data/facebookpages.json', 'rb'))

test_data1 = pd.DataFrame(test_json1)
test_data2 = pd.DataFrame(test_json2)

test_data1[0]
test_data2
sample outputs
['insurance plans for #healthcare companies will be on the rise for many years. this is why we need a #healthtech system for all. https://t.co/fkzvjkqzfk https://t.co/',
 'insurance companies can use GoodRx as a product, but they are not a pharmacy.  they are a pharmacy.  the cost of a pharmacy is more than a pharmacy means.  the price of a pharmacy is more than a prescription.  the',
 'insurance for the holidays is not a good choice. https://t.co/qkzfkqzgvq!!!!!!!!!!!!!!!!!!!!!!!!!',
 "insurance costs have been a problem since 2020. it's time to fight the #healthcare industry. #healthcare #hcldr https://t.co/qhxwzqxzfk!!!!!!!",
 'insurance costs for patients are over $10,000, but a new blog by @sabcsm https://t.co/xwvzgwjzgj via @youtube!!!!!!!!!!!']
