import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import bot_utils as butils
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
# %%

tokenizer=GPT2Tokenizer.from_pretrained('gpt2')

training_set_path = '../data/tweets_topics.csv'
data = pd.read_csv(training_set_path)
preprocessor = butils.Comment_data_preprocessor(data, 'id', 'text', tokenizer, keyword_field = 'topics')
tokenized_comments = preprocessor.df_to_tokenized_df(number_of_keywords = None)



parameter_dict = {}
#Currently huggingface defaults for training GPT2 (except more epochs)
parameter_dict['dataset_choice'] = 1
parameter_dict['model_choice'] = 1
parameter_dict['training_set_path'] = training_set_path
parameter_dict['epochs'] = 5
parameter_dict['num_worker'] = 2
parameter_dict['batch_size'] =2
parameter_dict['learning_rate'] =5e-5
parameter_dict['weight_decay'] = 0
parameter_dict['eps'] =1e-8
parameter_dict['warmup_steps'] =0
parameter_dict['filename'] ='gpt2_bagofwordskeywords_all_keywords_042220'

results_dir ='../results'
model_storage_dir ='../saved_models'

results_path = Path(Path(results_dir)/Path(parameter_dict['filename']))
model_path = Path(Path(model_storage_dir)/Path(parameter_dict['filename']))

results_path.mkdir(parents = True, exist_ok = True)
model_path.mkdir(parents = True, exist_ok = True)

if parameter_dict['dataset_choice'] == 0:
    dataset = prepend_ctrl_Dataset(preprocessor)

if parameter_dict['dataset_choice'] == 1:
    dataset = bag_words_ctrl_Dataset(preprocessor)

if parameter_dict['model_choice'] == 0:
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    trained_model, optimizer, scheduler, loss_data = butils.train(dataset,
                                                                  parameter_dict['epochs'],
                                                                  parameter_dict['num_worker'],
                                                                  parameter_dict['batch_size'],
                                                                  parameter_dict['learning_rate'],
                                                                  parameter_dict['weight_decay'],
                                                                  parameter_dict['eps'],
                                                                  parameter_dict['warmup_steps'],
                                                                  model,
                                                                  collate_fn = dataset.collate
                                                                  )

if parameter_dict['model_choice'] == 1:
    model = models.GPT2Model_bagofctrl.from_pretrained('gpt2')

    trained_model, optimizer, scheduler, loss_data = butils.train_bag_of_words(dataset,
                                                                  parameter_dict['epochs'],
                                                                  parameter_dict['num_worker'],
                                                                  parameter_dict['batch_size'],
                                                                  parameter_dict['learning_rate'],
                                                                  parameter_dict['weight_decay'],
                                                                  parameter_dict['eps'],
                                                                  parameter_dict['warmup_steps'],
                                                                  model,
                                                                  collate_fn = dataset.collate)
#saving
tokenized_comments.to_csv(results_path/'training_data.csv')
model.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
tokenizer.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
model.config.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
torch.save(optimizer.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' optimizer'))
torch.save(scheduler.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' scheduler'))

with open(results_path/'parameters.json', 'w') as jsonFile:
    json.dump(parameter_dict, jsonFile)

np.savetxt(results_path/'loss_data', loss_data, delimiter = ',')

#plotting
plt.clf()
plt.plot(range(parameter_dict['epochs']), loss_data)
plt.savefig(results_dir + '/' + parameter_dict['filename'] +'/'+'loss_plot.png')
