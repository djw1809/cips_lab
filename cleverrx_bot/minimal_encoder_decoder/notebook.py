import pickle
from transformers import BertTokenizer, EncoderDecoderModel, TrainingArguments, Trainer
import transformers
#import nlp
import datasets
import json
#%% Loading Data
# with open('../data/pairs_v3.pkl', 'rb') as file:
#     data = pickle.load(file)
#
#
# len(data)
# split = .9 * 549800
#
# example = data[0]
# example
# shorter_data = data[0:10000]
# example_tweet = example['tweet']
# example_tweet
# another_example = 'this is some more text to play with '
#
# train = data[0:494820]
# test = data[494820:len(data)]
# train = {'data':train}
# test = {'data':test}
#
# with open('pairs_v3_train.json', 'w') as file:
#     json.dump(train, file)# with open('../data/pairs_v3.pkl', 'rb') as file:
#     data = pickle.load(file)
#
#
# len(data)
# split = .9 * 549800
#
# example = data[0]
# example
# shorter_data = data[0:10000]
# example_tweet = example['tweet']
# example_tweet
# another_example = 'this is some more text to play with '
#
# train = data[0:494820]
# test = data[494820:len(data)]
# train = {'data':train}
# test = {'data':test}
#
# with open('pairs_v3_train.json', 'w') as file:
#     json.dump(train, file)
#
# with open('pairs_v3_tesy.json', 'w') as file:
#     json.dump(test, file)
#
# with open('pairs_v3_tesy.json', 'w') as file:
#     json.dump(test, file)




#dataset = datasets.load_dataset('json', data_files = {'train':'pairs_v3_train.json', 'test':'pairs_v3_tesy.json'}, field = 'data', split = 'train')
#dataset.column_names #returns just the train set as a huggingface dataset object
#dataset = datasets.load_dataset('json', data_files = {'train':'pairs_v3_train.json', 'test':'pairs_v3_tesy.json'}, field = 'data') returns a dict of huggingface dataset objects where the keys are they keys in the datafiles object
with open('../data/pairs_v3.pkl', 'rb') as file:
    data = pickle.load(file)


data = [i for i in data if i['rank'] == 1]


data = {"data": data}
with open('../data/pairs_v3.json', 'w') as file:
    json.dump(data, file)


dataset = datasets.load_dataset('json', data_files = '../data/pairs_v3.json', field = 'data')['train']
#%%

#%% Load model and tokenizer
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

#%% set decoding params all of these will be used as defaults when calling model.generate
model.config.decoder_start_token_id = tokenizer.bos_token_id #note that this doesnt affect training - only a convenient thing to use for generation
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142 # this is the max generation length NOT the max length of model inputs
model.early_stopping = True #a parameter for beam search - defined in GenerationMixin class
model.config.no_repeat_ngram_size = 3
model.length_penalty = 2.0
model.num_beams = 4 # using beam search by default

#%% write a mapping function for the dataset, we want to produce tweets in the style of the fb_posts so input to encoder should be fb_posts and input to decoder should be tweets
def map_to_encoder_decoder_inputs(batch):
    inputs = tokenizer(batch['fb_post'], padding = "max_length", truncation = True, max_length = 512)
    outputs = tokenizer(batch['tweet'], padding = "max_length", truncation = True, max_length = 512)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()

    #mask loss for padding --> Why do  I need to do this?
    batch["labels"] = [ [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"] ]
    batch["decoder_attention_mask"] = outputs.attention_mask

    return batch

#prepare for training, Q: What does predict from generate do?
batch_size = 2
dataset = dataset.map(map_to_encoder_decoder_inputs, batched = True, batch_size = batch_size, remove_columns = ["fb_post", "distance", "tweet", "rank"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],)

training_args = TrainingArguments(
                    output_dir = '../saved_models/minimal_encoder_decoder_rank1',
                    per_device_train_batch_size = batch_size,
                    predict_from_generate=True,
                    evaluate_during_training = False,
                    do_train = True,
                    do_eval = False,
                    warmup_steps = 1000,
                    save_total_limit=10,)

trainer = Trainer(model = model,
                    args = training_args,
                    train_dataset = dataset)


trainer.train()
