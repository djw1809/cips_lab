import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from ast import literal_eval


class Comment_pair_dataset(Dataset):

    def __init__(self, raw_data, tokenizer, sample_column1 = None, sample_column2 = None, already_tokenized = False):

        self.raw_data = raw_data
        self.input_data = pd.DataFrame(raw_data)
        self.get_type = 'sample1_first'
        self.max_len = None
        self.tokenizer = tokenizer
        if already_tokenized:
            self.active_data = self.input_data
        else:
            self.sample_column1 = sample_column1
            self.sample_column2 = sample_column2
            self.active_data = self.df_to_tokenized_df(self.input_data, self.sample_column1, self.sample_column2)

    def df_to_tokenized_df(self, input_data, sample_column1, sample_column2):
        tokenized_df = pd.DataFrame(columns = ['id', 'sample1', 'sample2', 'tokenized_text_sample1', 'tokenized_text_sample2', 'token_ids_sample1', 'token_ids_sample2'])
        tokenized_df.loc[:, 'id'] = input_data.index
        tokenized_df.loc[:, 'sample1'] = input_data.loc[:, sample_column1]
        tokenized_df.loc[:, 'sample2'] = input_data.loc[:, sample_column2]
        tokenized_df.loc[:, 'tokenized_text_sample1'] = input_data.loc[:, sample_column1].apply(self.tokenizer.tokenize)
        tokenized_df.loc[:, 'tokenized_text_sample2'] = input_data.loc[:, sample_column2].apply(self.tokenizer.tokenize)
        tokenized_df.loc[:, 'token_ids_sample1'] = input_data.loc[:, sample_column1].apply(self.tokenizer.encode)
        tokenized_df.loc[:, 'token_ids_sample2'] = input_data.loc[:, sample_column2].apply(self.tokenizer.encode)

        return tokenized_df

    def set_get_type(self,type):
        if type == 'sample1_first':
            self.get_type = 'sample1_first'

        elif type == 'sample2_first':
            self.get_type = 'sample2_first'

        else:
            print('not a valid get type')


    def __getitem__(self, index):

        if self.max_len is None:
            sample1 = self.active_data.loc[index, 'token_ids_sample1']
            sample2 = self.active_data.loc[index, 'token_ids_sample2']

        else:
            sample1 = self.active_data.loc[index, 'token_ids_sample1'][:self.max_len]
            sample2 = self.active_data.loc[index, 'token_ids_sample2'][:self.max_len]


        if self.get_type == 'sample1_first':
            return (sample1, sample2)

        elif self.get_type == 'sample2_first':
            return (sample2, sample1)

    def __len__(self):
        return len(self.active_data)

    def collate(self, batch):
        tokenizer = self.tokenizer
        sample1_ids = [torch.tensor(item[0]) for item in batch]
        sample2_ids = [torch.tensor(item[1]) for item in batch]

        if tokenizer._pad_token is None:
             padded_sample1_ids = pad_sequence(sample1_ids, batch_first = True)
             padded_sample2_ids = pad_sequence(sample2_ids, batch_first = True)
        else:
             padded_sample1_ids = pad_sequence(sample1_ids, batch_first = True, padding_value = tokenizer.pad_token_id)
             padded_sample2_ids = pad_sequence(sample2_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_sample1_ids, padded_sample2_ids



def train_hugging_encode_decode_keyword(training_dataset, epochs, num_workers, batch_size, learning_rate, weight_decay, eps, warmup_steps, model, collate_fn = None):
    '''generic training call for a pytorch model'''


    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, collate_fn = collate_fn)


#### configure model to use cuda if it is available ####
    print("CUDA available is {}".format(torch.cuda.is_avilable()))
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
