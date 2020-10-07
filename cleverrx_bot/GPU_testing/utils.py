import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from ast import literal_eval


class Comment_data_preprocessor(Dataset):
    '''class to do different things quickly with raw_data
        -properties: raw_data, input_df, train_df (the dataframe used for training), test_df(the dataframe used for testing), current_df(either test or train, the df that will be accessed by get_item), tokenizer, corpus
        - always assume input has text field, id field
        - TODO: assert that if a synonym dict has been provided then so should a keyword_field
        '''
    #PREPROCESSING
    def __init__(self, raw_data, text_field, tokenizer, id_field = None, keyword_field = None, synonym_dict = None):
        #sefl.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.synonym_dict = synonym_dict
        self.raw_data = raw_data
        self.corpus = None
        self.input_df = None
        self.prepared_datasets = {}
        self.active_dataset = None # This is the dataset that will be accessed by __getitem__ and __len__
        self.get_type = 'keyword'  #keyword: return a tuple of (tokenized keywords, tokenized text), prepend: return tokenized keyword prepended text
        self.collate_fn = self.collate_keyword

        if type(raw_data) == str: #process input data if it is a corpus
            self.corpus = raw_data
            self.input_df = self.text_to_chunked_df(self.raw_text)

        elif isinstance(raw_data, pd.DataFrame): #process input data if it is a dataframe
            self.input_df = raw_data
            if id_field is not None:
                self.input_df = self.input_df.rename(columns = {text_field: 'text', id_field: 'id'})

            else:
                self.input_df = self.input_df.rename(columns = {text_field: 'text'})
                self.input_df['id'] = range(len(self.input_df))
            # intermediate_df = pd.DataFrame(columns = ['id', 'text'])
            # intermediate_df.loc[:, 'id'] = raw_data.loc[:, id_field]
            # intermediate_df.loc[:, 'text'] = raw_data.loc[:, text_field]
            # if keyword_field != None:
            #     intermediate_df.loc[:, 'keywords'] = raw_data.loc[:, keyword_field]
            #     intermediate_df.loc[:, 'keywords'] = intermediate_df.loc[:, 'keywords'].apply(literal_eval) #make sure keywords are in a list


        elif isinstance(raw_data, list):  #process input data if it is json list of dicts, each dict representing a comment
            self.input_df = pd.DataFrame(raw_data)

            if id_field is not None:
                self.input_df = self.input_df.rename(columns = {text_field: 'text', id_field: 'id'})

            else:
                self.input_df = self.input_df.rename(columns = {text_field: 'text'})
                self.input_df['id'] = range(len(self.input_df))

            # if keyword_field != None:
            #     drop_columns = [column for column in intermediate_df.columns if column not in [id_field, text_field, keyword_field]]
            # else:
            #     drop_columns = [column for column in intermediate_df.columns if column not in [id_field, text_field]]
            #
            # intermediate_df = intermediate_df.drop(columns = drop_columns)
            #
            # if keyword_field != None:
            #     intermediate_df = intermediate_df.rename(columns = {id_field: 'id', text_field: 'text', keyword_field: 'keywords'})
            #     intermediate_df.loc[:, 'keywords'] = intermediate_df.loc[:, 'keywords'].apply(literal_eval) #make sure keywords are in a list
            # else:
            #     intermediate_df = intermediate_df.rename(columns = {id_field: 'id', text_field: 'text'})

        elif isinstance(raw_data, dict): #process input data if it is json dict of dicts, each dict representing a comment with ids as keys
            self.input_df = pd.DataFrame.from_dict(raw_data, orient = 'index')
            self.input_df['id'] = self.input_df.index
            self.input_df = self.input_df.rename(columns = {text_field: 'text'})


        else:
            pass

        self.input_df = self.input_df.dropna(subset = ['text']) #drop rows where there is no text
        self.input_df.index = range(len(self.input_df))



def train_hugging_encode_decode_keyword(training_dataset, epochs, num_workers, batch_size, learning_rate, weight_decay, eps, warmup_steps, model, collate_fn = None):
    '''generic training call for a pytorch model'''


    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, collate_fn = collate_fn)


#### configure model to use cuda if it is available ####
    print("CUDA available is {}".format(torch.cuda.is_available()))
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
    epoch_counter = 0
    for epoch in range(epochs):
        batch_counter = 0

        running_loss = 0
        model.train()

        for batch  in training_loader:
            #forwards
            print('epoch {} batch {}'.format(epoch_counter, batch_counter))
            print('Memory allocated {}'.format(torch.cuda.memory_allocated(device = device)))
            texts, keywords = batch
            decoder_input_ids = texts
            input_ids = keywords
            input_ids = torch.LongTensor(input_ids)
            decoder_input_ids = torch.LongTensor(decoder_input_ids)
            decoder_input_ids = decoder_input_ids.to(device)
            input_ids = input_ids.to(device)
            outputs = model(input_ids = input_ids, decoder_input_ids = decoder_input_ids, labels = decoder_input_ids)
            loss = outputs[0]

            #backwards
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_loss += loss.item()
            batch_counter += 1
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
        epoch_counter += 1
        print(' Loss: {:.4f} '.format(epoch_loss))

    return model, optimizer, scheduler, loss_data
