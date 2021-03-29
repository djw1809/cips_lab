import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Model, get_linear_schedule_with_warmup
from ast import literal_eval
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
#%%


#%%
class GPT2Model_bagofctrl(GPT2Model):
    def __init__(self, config, n_embed = 768):
        super().__init__(config)
        self.n_embed = n_embed
        self.lm_head = nn.Linear(self.n_embed, config.vocab_size, bias = False)
        self.tokenizer = None

        self.init_weights()

    def forward(self,
                batch,
                device, #(sequence ids - batchsize x seqlen, list of keywords - list of lenth batchsize, each list entry is a tensor of keywords)
                position_ids = None,
                labels = None):



        input_ids, keyword_ids = batch
        batch_size = len(keyword_ids)
        input_embedding = self.wte(input_ids) # batchsize x seqlen x embed_dim
        keyword_embedding = torch.zeros(batch_size, self.wte.embedding_dim)

        #set up keyword embeddings
        for i in range(len(keyword_ids)):
            keyword_list = keyword_ids[i]
            if len(keyword_list) == 0: #for training/generation with no keywords
                keyword_embedding[i, :] = torch.rand(self.wte.embedding_dim)
            else:
                keyword_embeddings = self.wte(keyword_list)
                bag_of_words = torch.mean(keyword_embeddings, 0)
                keyword_embedding[i, :] = bag_of_words

        keyword_embedding = keyword_embedding.unsqueeze(1).to(device)
        final_embedding = torch.cat((keyword_embedding, input_embedding),1)

        #set up posistional embeddings
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(0, final_embedding.shape[1], dtype = torch.long, device = device)
            position_ids = position_ids.unsqueeze(0).view(-1, final_embedding.shape[1])

        #Compute/reshape input
        position_embedding = self.wpe(position_ids)
        hidden_states = final_embedding + position_embedding
        hidden_states = self.drop(hidden_states)
        output_shape = hidden_states.shape

        #compute output hidden state (output of attention layers)
        for i, block in enumerate(self.h): #
            hidden_states = block(hidden_states)[0]

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        #compute language modeling objective
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits, hidden_states)

        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            #shift_labels = labels[:, 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


    def generate(self, tokenizer, prompt, max_length, top_k = None, top_p = None, num_return_sequences = 1, min_keep = 1, filter_value = -float("Inf")):
        '''generation with bag of words ctrl.  prompt should be of the form (list of keywords, start of generated sentence)'''

        #model = self.model
        #setup device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #if torch.cuda.is_available():
            #model.cuda()
        self.eval()

        #encode prompt
        if len(prompt) == 1:
            keywords,  = prompt

        else:
            keywords, bos = prompt
            bos_tokens = tokenizer.encode(bos)

        keyword_tokens = []
        if len(keywords) > 0:
            for keyword in keywords:
                keyword_tokens = keyword_tokens + tokenizer.encode(keywords)

            keyword_tokens = [torch.tensor(keyword_tokens)] #put things in right shape for forward pass
        else:
            keyword_tokens = [[]]

        returned_sequences = []

        for i in range(num_return_sequences):
            if len(prompt) > 1:
                sequence_tokens = bos_tokens
            else:
                sequence_tokens = []
            for j in range(max_length):
                #obtain logits
                input_ids = torch.tensor(sequence_tokens, dtype = torch.long).unsqueeze(0)
                logits = self.forward((input_ids, keyword_tokens), device)[0][:, -1, :] #get logits of the predicted word

            #perform top_k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][:,-1, None] #return the indicies
                    logits[indices_to_remove] = filter_value  #mask the bad ones
            #perform "nucleus" sampling
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1 ), dim = -1)
                    sorted_indices_to_remove = cum_probs > top_p
                    if min_keep > 1:
                        sorted_indices_to_remove[:, :min_keep] = 0

                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() #shift everything to right - will always pick first token above threshhold as well now
                    sorted_indices_to_remove[:, 0] = 0  #always keep at least most probable

                    #put everything in the right place
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

                    logits[indices_to_remove] = filter_value

                if top_k > 0 or top_p > 0:
                    next_token_index = [int(torch.multinomial(F.softmax(logits), 1))]
                    #print('finished token {}'.format(j))
                else:
                    next_token_index = [int(torch.argmax(logits))]

                sequence_tokens = sequence_tokens + next_token_index

            returned_sequences.append(sequence_tokens)

        returned_sentences = []

        for sequence in returned_sequences:
            decoded_sequence = tokenizer.decode(sequence, clean_up_tokenization_spaces = True)
            returned_sentences.append(decoded_sequence)

        return returned_sentences, returned_sequences

    def set_tokenizer(tokenizer):
        '''if the model was trained using a specific tokenizer we can set it as an attribute of the model class instance'''
        self.tokenizer = tokenizer

    @classmethod #so that I can load a bag of ctrl model with loaded_model = GPT2Model_bagofctrl.load('place where I saved everything')
    def load(cls, path_to_results_folder):
        model = super().from_pretrained(path_to_results_folder)
        return model
