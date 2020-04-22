# %%
import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from ast import literal_eval
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class Comment_data_preprocessor():
    '''class to do different things quickly with raw_data
        -properties: raw_data, input_df, train_df (the dataframe used for training), test_df(the dataframe used for testing), current_df(either test or train, the df that will be accessed by get_item), tokenizer, corpus
        - always assume input has text and id field (if json or df)
        - TODO: assert that if a synonym dict has been provided then so should a keyword_field
        '''
    def __init__(self, raw_data, id_field, text_field, tokenizer, keyword_field = None, synonym_dict = None):
        #sefl.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.synonym_dict = synonym_dict
        self.raw_data = raw_data
        self.corpus = None
        self.input_df = None
        self.tokenized_df = None
        self.train_df = None
        self.test_df = None

        if type(raw_data) == str: #process input data if it is a corpus
            self.corpus = raw_data
            self.input_df = self.text_to_chunked_df(self.raw_text)

        elif isinstance(raw_data, pd.DataFrame): #process input data if it is a dataframe
            intermediate_df = pd.DataFrame(columns = ['id', 'text'])
            intermediate_df.loc[:, 'id'] = raw_data.loc[:, id_field]
            intermediate_df.loc[:, 'text'] = raw_data.loc[:, text_field]
            if keyword_field != None:
                intermediate_df.loc[:, 'keywords'] = raw_data.loc[:, keyword_field]
                intermediate_df.loc[:, 'keywords'] = intermediate_df.loc[:, 'keywords'].apply(literal_eval) #make sure keywords are in a list


        elif isinstance(raw_data, list):  #process input data if it is json should be a list of dicts, each dict representing a comment
            intermediate_df = pd.DataFrame(raw_data)
            if keyword_field != None:
                drop_columns = [column for column in intermediate_df.columns if column not in [id_field, text_field, keyword_field]]
            else:
                drop_columns = [column for column in intermediate_df.columns if column not in [id_field, text_field]]

            intermediate_df = intermediate_df.drop(columns = drop_columns)

            if keyword_field != None:
                intermediate_df = intermediate_df.rename(columns = {id_field: 'id', text_field: 'text', keyword_field: 'keywords'})
                intermediate_df.loc[:, 'keywords'] = intermediate_df.loc[:, 'keywords'].apply(literal_eval) #make sure keywords are in a list
            else:
                intermediate_df = intermediate_df.rename(columns = {id_field: 'id', text_field: 'text'})



        else:
            pass

        intermediate_df = intermediate_df.dropna(subset = ['text']) #drop rows where there is no text
        intermediate_df.index = range(len(intermediate_df))
        self.input_df = intermediate_df



    def df_to_corpus(self):
        self.corpus = ''
        for i in range(len(self.input_df)):
            self.corpus = self.corpus + ' ' + self.input_df.loc[i, 'text']

    def tokenize_list_of_keywords(self, input_list):
        output_tokens = []
        for keyword in input_list:
            output_tokens = output_tokens + self.tokenizer.tokenize(keyword)

        return output_tokens

    def encode_list_of_keywords(self, input_list):
        output_ids = []
        for keyword in input_list:
            output_ids = output_ids + self.tokenizer.encode(keyword)

        return output_ids


    def df_to_tokenized_df(self, number_of_keywords = None):

        if 'keywords' in self.input_df.columns:
            self.tokenized_df = pd.DataFrame(columns = ['id', 'text', 'raw_keywords', 'tokenized_text', 'tokenized_keywords', 'text_ids', 'keyword_ids'])
            self.tokenized_df.loc[:,'id'] = self.input_df.loc[:,'id']
            self.tokenized_df.loc[:,'text'] = self.input_df.loc[:,'text']
            self.tokenized_df.loc[:, 'tokenized_text'] = self.tokenized_df.loc[:, 'text'].apply(self.tokenizer.tokenize)
            self.tokenized_df.loc[:, 'text_ids'] = self.tokenized_df.loc[:, 'text'].apply(self.tokenizer.encode)
            self.tokenized_df.loc[:,'raw_keywords'] = self.input_df.loc[:,'keywords']
            self.tokenized_df.index = self.input_df.index
            self.tokenized_df.loc[:,'used_keywords'] = np.nan
            self.tokenized_df.loc[:,'used_keywords'] = self.tokenized_df.loc[:,'used_keywords'].astype('object')

            if self.synonym_dict != None:

                for row in self.tokenized_df.index:

                    keywords = self.input_df.loc[row, 'keywords'].copy()
                    translated_keywords = []

                    if number_of_keywords != None:
                        translate_range = number_of_keywords
                    else: #if no number is given translate all of them
                        translate_range = len(keywords)

                    for i in range(translate_range):
                        if len(keywords) == 0: #incase there are examples where the number of keywords is less than the translate range
                            break
                        else:
                            keyword = keywords.pop()
                            try:
                                translated_keywords.append(self.synonym_dict[keyword])
                            except KeyError: ##if there is no translation for the keyword just keep it
                                translated_keywords.append(keyword)

                    translated_keywords = list(set(translated_keywords)) #remove duplicates
                    self.tokenized_df.at[row, 'used_keywords'] = translated_keywords






            else:
                for row in self.tokenized_df.index:
                    keywords = list(self.input_df.loc[row, 'keywords'])
                    prepended_keywords = []
                    if number_of_keywords != None:
                        translate_range = number_of_keywords
                    else:
                        translate_range = len(keywords)
                        for i in range(translate_range):
                            if len(keywords) == 0:
                                break
                            else:
                                keyword = keywords.pop()
                                prepended_keywords.append(keyword)
                                self.tokenized_df.at[row, 'used_keywords'] = prepended_keywords


            self.tokenized_df.loc[:, 'tokenized_keywords'] = self.tokenized_df.loc[:, 'used_keywords'].apply(self.tokenize_list_of_keywords)
            self.tokenized_df.loc[:, 'keyword_ids'] = self.tokenized_df.loc[:, 'used_keywords'].apply(self.encode_list_of_keywords)


        else:
            self.tokenized_df = pd.DataFrame(columns = ['id', 'text', 'tokenized_text', 'token_ids'])
            self.tokenized_df.loc[:,'id'] = self.input_df.loc[:, 'id']
            self.tokenized_df.loc[:, 'text'] = self.input_df.loc[:, 'text']
            self.tokenized_df.loc[:, 'tokenized_text'] = self.input_df.loc[:, 'text'].apply(self.tokenizer.tokenize)
            self.tokenized_text.loc[:, 'token_ids'] = self.input_df.loc[:, 'text'].apply(self.tokenizer.encode)

        return self.tokenized_df





class Comment_dataset(Dataset):

    def __init__(self, raw_data, sample_column):
        self.sample_column = sample_column
        self.data = raw_data

    def __getitem__(self, index):
        return torch.tensor(self.data.loc[index, self.sample_column], dtype = torch.long)

    def __len__(self):
        return len(self.data)

class prepend_ctrl_Dataset(Dataset):

    def __init__(self, preprocessor, tokenizer = None):
        if preprocessor.tokenized_df is None:
            preprocessor.tokenized_df()

        self.data = preprocessor.tokenized_df

        if tokenizer != None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = preprocessor.tokenizer
            except:
                print("no tokenizer found - collate function wont work")


    def __getitem__(self, index):
        text_ids = self.data.loc[index, 'text_ids']
        keyword_ids = self.data.loc[index,'keyword_ids']
        prepended_ids = keyword_ids + text_ids
        return prepended_ids

    def __len__(self):
        return len(self.data)


    def collate(self, batch):
        tokenizer = self.tokenizer
        text_ids = [torch.tensor(item) for item in batch]

        if tokenizer._pad_token is None:
             padded_texts = pad_sequence(text_ids, batch_first = True)
        else:
             padded_texts = pad_sequence(text_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_texts


class bag_words_ctrl_Dataset(Dataset):

    def __init__(self, preprocessor, tokenizer = None):
        if preprocessor.tokenized_df is None:
            preprocessor.tokenized_df()

        self.data = preprocessor.tokenized_df
        if tokenizer != None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = preprocessor.tokenizer
            except:
                print("no tokenizer found - collate function wont work")

    def __getitem__(self, index):
        text_ids = self.data.loc[index, 'text_ids']
        keyword_ids = self.data.loc[index, 'keyword_ids']
        return (text_ids, keyword_ids)

    def __len__(self):
        return len(self.data)


    def collate(self, batch):
        tokenizer = self.tokenizer
        text_ids = [torch.tensor(item[0]) for item in batch]
        keyword_ids = [torch.tensor(item[1]) for item in batch]

        if tokenizer._pad_token is None:
             padded_texts = pad_sequence(text_ids, batch_first = True)
        else:
             padded_texts = pad_sequence(text_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_texts, keyword_ids










# %%


def train(training_dataset, epochs, num_workers, batch_size, learning_rate, weight_decay, eps, warmup_steps, model, collate_fn = None):
    '''generic training call for a pytorch model'''


    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, collate_fn = collate_fn)


#### configure model to use cuda if it is available ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()

#### initialize containers to store model outputs in ####
    loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in

### initialize optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] #set weight decay to 0 for bias and layernorm weights

    optimizer = AdamW(optimizer_grouped_parameters, lr= learning_rate, eps= eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps= len(training_loader)
    )

##### MAIN TRAINING LOOP ######

    for epoch in range(epochs):

        running_loss = 0
        model.train()

        for batch  in training_loader:
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            #optimizer.zero_grad()

            #forward
            loss = model(inputs, labels = labels)[0]

            #backwards
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_loss += loss.item()
            #running_corrects += torch.sum(preds == labels.data).item()
            #confusion_matrix_train_epoch += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels =range(num_labels))

##### calculate  epoch data ####
        epoch_loss = running_loss / len(training_dataset)
        #epoch_corrects = running_corrects / len(training_dataset)
        #epoch_val_accuracy = running_val_corrects/len(test_dataset)

###### record epoch data ###########
        loss_data[epoch] = epoch_loss
        #accuracy_data[epoch] = epoch_corrects
        #val_accuracy_data[epoch] = epoch_val_accuracy
        #confusion_matricies_test[epoch] = confusion_matrix_test_epoch
        #confusion_matricies_train[epoch] = confusion_matrix_train_epoch

        print(' Loss: {:.4f} '.format(epoch_loss))

    return model, optimizer, scheduler, loss_data


def train_bag_of_words(training_dataset, epochs, num_workers, batch_size, learning_rate, weight_decay, eps, warmup_steps, model, collate_fn = None):
    '''generic training call for a pytorch model'''


    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, collate_fn = collate_fn)


#### configure model to use cuda if it is available ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()

#### initialize containers to store model outputs in ####
    loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in

### initialize optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] #set weight decay to 0 for bias and layernorm weights

    optimizer = AdamW(optimizer_grouped_parameters, lr= learning_rate, eps= eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps= len(training_loader)
    )

##### MAIN TRAINING LOOP ######

    for epoch in range(epochs):

        running_loss = 0
        model.train()

        for batch  in training_loader:
            inputs, labels = (batch, batch[0])
            device_sequence = inputs[0].to(device)
            device_keywords = [inputs[1][i].to(device) for i in range(len(inputs[1]))]
            inputs = (device_sequence, device_keywords)
            labels = labels.to(device)
            #optimizer.zero_grad()

            #forward
            loss = model(inputs, device, labels = labels)[0]

            #backwards
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_loss += loss.item()
            #running_corrects += torch.sum(preds == labels.data).item()
            #confusion_matrix_train_epoch += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels =range(num_labels))

##### calculate  epoch data ####
        epoch_loss = running_loss / len(training_dataset)
        #epoch_corrects = running_corrects / len(training_dataset)
        #epoch_val_accuracy = running_val_corrects/len(test_dataset)

###### record epoch data ###########
        loss_data[epoch] = epoch_loss
        #accuracy_data[epoch] = epoch_corrects
        #val_accuracy_data[epoch] = epoch_val_accuracy
        #confusion_matricies_test[epoch] = confusion_matrix_test_epoch
        #confusion_matricies_train[epoch] = confusion_matrix_train_epoch

        print(' Loss: {:.4f} '.format(epoch_loss))

    return model, optimizer, scheduler, loss_data

# %%


def generate_(model, tokenizer, prompt, max_length, do_sample = True, num_beams = None, temperature = None, top_k = None, top_p = None, repetition_penalty = None, num_return_sequences = 1,   print_ = True, stop_token = None):

    encoded_prompt = tokenizer.encode(prompt, add_special_tokens = False, return_tensors = "pt")
    output_sequences = model.generate(input_ids = encoded_prompt,
                                      max_length = max_length,
                                      temperature = temperature,
                                      top_k = top_k,
                                      top_p = top_p,
                                      repetition_penalty = repetition_penalty,
                                      do_sample = True,
                                      num_return_sequences = num_return_sequences)

    if len(output_sequences.shape) > 2:
        output_sequences = output_sequences.squeeze()

    generated_sequences = []

    for id, sequence in enumerate(output_sequences):
        decoded_sequence = tokenizer.decode(sequence.tolist(), clean_up_tokenization_spaces = True)
        if stop_token != None:
            decoded_sequence = decoded_sequence[: (decoded_sequence.find(stop_token) + 1)]

        if print_:
            print("Generated sequence {}: {}".format(id, prompt + ' ' + decoded_sequence))

        generated_sequences.append(prompt + ' ' + decoded_sequence)


    return output_sequences
