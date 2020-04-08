import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import bot_utils as butils
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path

tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
#synonym_dict =
data = pd.read_csv('../data/tweets_topics.csv')
data = data.loc[0:100, :]
pre_processor = butils.Comment_data_preprocessor(data, 'id', 'text', tokenizer, 'topics')
tokenized_comments = pre_processor.df_to_tokenized_df(number_of_keywords = 1)
dataset = butils.Comment_dataset(tokenized_comments, 'token_ids')

parameter_dict = {}


parameter_dict['training_set'] = tokenized_comments
parameter_dict['epochs'] = 2
parameter_dict['num_worker'] = 1
parameter_dict['batch_size'] =1
parameter_dict['learning rate'] =1e-5
parameter_dict['weight_decay'] = 0
parameter_dict['eps'] =1e-8
parameter_dict['warmup_steps'] =100
parameter_dict['filenames'] ='test'

results_dir ='../results'
model_storage_dir ='../saved_models'

results_path = Path(results_dir/parameter_dict['filenames'])
model_path = Path(results_dir/parameter_dict['filenames'])

results_path.mkdir(parents = True, exist_ok = True)
model_path.mkdir(parents = True, exist_ok = True)

model = GPT2LMHeadModel.from_pretrained('gpt2')

trained_model, optimizer, scheduler, loss_data = butils.train(dataset, tokenizer,
                                                              parameter_dict['epochs'],
                                                              parameter_dict['num_worker'],
                                                              parameter_dict['batch_size'],
                                                              parameter_dict['learning_rate'],
                                                              parameter_dict['weight_decay'],
                                                              parameter_dict['eps'],
                                                              parameter_dict['warmup_steps'])
#saving
tokenized_comments.to_csv(results_path/'training_data.csv')
model.save_pretrained(model_path/parameter_dict['filename'] + 'model')
torch.save(optimizer.state_dict, model_path/parameter_dict['filename']+'optimizer')
torch.save(scheduler.state_dict, model_path/parameter_dict['filename']+'scheduler')

with open(results_path/'parameters.json') as jsonFile:
    json.dump(parameter_dict, jsonFile)

np.savetxt(results_path/'loss_data', loss_data, delimiter = ',')

#plotting
plt.clf()
plt.plot(range(parameter_dict['epoch']), loss_data)
plt.savefig(results_dir + '/' + parameter_dict['filenames'] +'/'+'loss_plot.png')

#plotting
plt.clf()
plt.plot(range(epochs), loss_data, 'bo')
plt.savefig(results_dir+'/'+parameter_dict['filenames']+'/'+'loss_plot.png')
