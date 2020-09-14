import pandas as pd
import numpy as np
import torch
import matplotlib
import faulthandler 
faulthandler.enable()
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, EncoderDecoderModel, BertTokenizer
from pathlib import Path
import torch.nn as nn
import bot_models as models
import bot_utils as butils
from bot_utils import Comment_data_preprocessor, Comment_dataset, Comment_pair_dataset


test = True
new_dataset = True
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
get_type = 'sample1_first'


if new_dataset:
    raw_data_path = '../data/pairs_v3.pkl'
    sample1_field = 'fb_post'
    sample2_field = 'tweet'
    json_ = False
    if json_:
        with open(raw_data_path, 'rb') as file:
            raw_data = json.load(file)
    else:
        with open(raw_data_path, 'rb') as file:
            raw_data = pickle.load(file)

    if test:
        raw_data = raw_data[0:5]

    dataset = Comment_pair_dataset(raw_data, sample1_field, sample2_field, tokenizer)
    dataset.set_get_type(get_type)
    dataset.max_len = 512 #tokenizer.max_len

else:
    data_path = '' #needs to be readable by pandas
    data_sample_column = 'token_ids'
    tokenized_df = pd.read_csv(data_path)
    dataset = Comment_pair_dataset(tokenized_df, sample1_field, sample2_field, tokenizer, already_tokenized = True)
    dataset.max_len = tokenizer.max_len

results_dir = '../results'
model_storage_dir = '../saved_models'


parameter_dict = {}
parameter_dict['epochs'] = 5
parameter_dict['num_worker'] = 2
parameter_dict['batch_size'] = 2
parameter_dict['learning_rate'] =5e-5
parameter_dict['weight_decay'] = 0
parameter_dict['eps'] =1e-8
parameter_dict['warmup_steps'] =0
parameter_dict['filename'] =  'pair_v3_encode_decode_091120_test'

results_path = Path(Path(results_dir)/Path(parameter_dict['filename']))
model_path = Path(Path(model_storage_dir)/Path(parameter_dict['filename']))
results_path.mkdir(parents = True, exist_ok = True)
model_path.mkdir(parents = True, exist_ok = True)

model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
trained_model, optimizer, scheduler, loss_data = butils.train_hugging_encode_decode_keyword(dataset, parameter_dict['epochs'],
                                                                                             parameter_dict['num_worker'],
                                                                                             parameter_dict['batch_size'],
                                                                                             parameter_dict['learning_rate'],
                                                                                             parameter_dict['weight_decay'],
                                                                                             parameter_dict['eps'],
                                                                                             parameter_dict['warmup_steps'],
                                                                                             model,
                                                                                             dataset.collate
                                                                                             )

dataset.active_data.to_csv(results_path/'training_data.csv')

trained_model.save_pretrained(model_storage_dir + '/' + parameter_dict['filename'])
tokenizer.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
trained_model.config.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])

#saving torch stuff - see torch docs for proper loading
torch.save(optimizer.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' optimizer'))
torch.save(scheduler.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' scheduler'))

#saving parameter dict
with open(results_path/'parameters.json', 'w') as jsonFile:
    json.dump(parameter_dict, jsonFile)

np.savetxt(results_path/'loss_data', loss_data, delimiter = ',')

#plotting
plt.clf()
plt.scatter(range(parameter_dict['epochs']), loss_data)
plt.savefig(results_dir + '/' + parameter_dict['filename'] +'/'+'loss_plot.png')
