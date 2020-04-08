# %%
import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from ast import literal_eval
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



# raw_data = pd.read_csv('../../data/new_topics.csv')
# raw_data
# with open('../../data/topics_index.pkl', 'rb') as file:
#     topics_index = pickle.load(file)
#
# better_data = pd.DataFrame(columns = ['id', 'text', 'topics'])
# for i in range(len(list(topics_index.keys()))):
#     key = list(topics_index.keys())[i]
#     tweet = topics_index[key]
#     text = tweet['tweet']
#     id = key
#     topics = list(set().union(tweet['cost_list'],
#                             tweet['card_list'],
#                             tweet['customers_list'],
#                             tweet['health_list'],
#                             tweet['inhaler_list'],
#                             tweet['insurance_list'],
#                             tweet['medication_list'],
#                             tweet['patients_list'],
#                             tweet['religion_list'],
#                             tweet['segment_list'],
#                             tweet['service_list'],
#                             tweet['transparency_list'],
#                             ))
#     row = {'id': id, 'text':text, 'topics':topics}
#     better_data.loc[i, :] = row
#
# better_data.to_csv('tweets_topics.csv')
#
# dataset = butils.Data_handler(better_data)
#
# print(dataset.raw_text)
# dataset.raw_df


###data processing###

# %%




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


    def df_to_tokenized_df(self, number_of_keywords = None):

        if 'keywords' in self.input_df.columns:
            self.tokenized_df = pd.DataFrame(columns = ['id', 'text', 'raw_keywords', 'prepended_text', 'prepended_tokenized_text', 'prepended_token_ids'])
            self.tokenized_df.loc[:,'id'] = self.input_df.loc[:,'id']
            self.tokenized_df.loc[:,'text'] = self.input_df.loc[:,'text']
            self.tokenized_df.loc[:,'raw_keywords'] = self.input_df.loc[:,'keywords']
            self.tokenized_df.index = self.input_df.index

            if self.synonym_dict != None:
                self.tokenized_df.loc[:,'translated_keywords'] = np.nan
                self.tokenized_df.loc[:,'translated_keywords'] = self.tokenized_df.loc[:,'translated_keywords'].astype('object')
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
                    self.tokenized_df.at[row, 'translated_keywords'] = translated_keywords

                    prepend_string = ''
                    for keyword in translated_keywords:
                        prepend_string += keyword + ' '
                    prepend_string = prepend_string.rstrip()
                    self.tokenized_df.loc[row, 'prepended_text'] = prepend_string + ' ' +self.tokenized_df.loc[row, 'text']



            else:
                for row in self.tokenized_df.index:
                    keywords = list(self.input_df.loc[row, 'keywords'])
                    prepended_keywords = []
                    if number_of_keywords != None:
                        translate_range = number_of_keywords
                        for i in range(translate_range):
                            if len(keywords) == 0:
                                break
                            else:
                                keyword = keywords.pop()
                                prepended_keywords.append(keyword)

                    prepend_string = ''
                    for keyword in prepended_keywords:
                        prepend_string  += keyword + ' '
                    prepend_string = prepend_string.rstrip()
                    self.tokenized_df.loc[row, 'prepended_text'] = prepend_string + ' ' + self.tokenized_df.loc[row, 'text']


            self.tokenized_df.loc[:, 'prepended_tokenized_text'] = self.tokenized_df.loc[:, 'prepended_text'].apply(self.tokenizer.tokenize)
            self.tokenized_df.loc[:, 'prepended_token_ids'] = self.tokenized_df.loc[:, 'prepended_text'].apply(self.tokenizer.encode)



        else:
            self.tokenized_df = pd.DataFrame(columns = ['id', 'text', 'tokenized_text', 'token_ids'])
            self.tokenized_df.loc[:,'id'] = self.input_df.loc[:, 'id']
            self.tokenized_df.loc[:, 'text'] = self.input_df.loc[:, 'text']
            self.tokenized_df.loc[:, 'tokenized_text'] = self.input_df.loc[:, 'text'].apply(self.tokenizer.tokenize)
            self.tokenized_text.loc[:, 'token_ids'] = self.input_df.loc[:, 'text'].apply(self.tokenizer.encode)

        return self.tokenized_df

    def collate(self, batch):
        if self.tokenizer._pad_token is None:
            return pad_sequence(batch, batch_first=True)
        return pad_sequence(batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)





class Comment_dataset(Dataset):

    def __init__(self, raw_data, sample_column):
        self.sample_column = sample_column
        self.data = raw_data

    def __getitem__(self, index):
        return torch.tensor(self.data.loc[index, self.sample_column], dtype = torch.long)

    def __len__(self):
        return len(self.data)






# %%


def train(training_dataset, tokenizer, epochs, num_workers, batch_size, learning_rate, weight_decay,eps,warmup_steps, model):
    '''generic training call for a pytorch model'''

    def collate(batch):
        if tokenizer._pad_token is None:
            return pad_sequence(batch, batch_first=True)
        return pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)


    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, collate_fn = collate)


#### configure model to use cuda if it is available ####

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model.cuda()
        weight = weight.to(device)
    loss = CrossEntropyLoss_weight(weight)

#### initialize containers to store model outputs in ####

    loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in
    accuracy_data = np.zeros((epochs))
    val_accuracy_data = np.zeros((epochs))

    confusion_matricies_test = {}
    confusion_matricies_train = {}

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

#### initialize epoch data #####

        running_loss = 0
        #running_corrects = 0
        #running_val_corrects = 0

# ###### EVALUATION STEP ########
#         model.eval()
        #
        # for inputs, labels  in test_loader:
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
        #     inputs = inputs.float()
        #     labels = labels.long()
        #     outputs = model(inputs)
        #     _, preds = torch.max(outputs.data, 1)
        #     running_val_corrects += torch.sum(preds == labels.data).item()
        #     confusion_matrix_test_epoch += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels =range(num_labels))

##### TRAINING STEP ########
        model.train()

        for batch  in training_loader:
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            #optimizer.zero_grad()

            #forward
            loss, outputs = model(inputs, labels = labels)
            loss.backward()

            #backwards
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()
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

###Dataset classes - meant to "feed" datasets from pandas dataframes in the right way

#class GPT2_language_modeling_trainset(Dataset):
    #'''byte-pair encoding of input sentences using pretrained tokenizer'''
