import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Model, get_linear_schedule_with_warmup
from ast import literal_eval
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
#%%


#%%
class GPT2Model_bagofctrl(GPT2Model):
    def __init__(self, config, n_embed = 768):
        super().__init__(config)
        self.n_embed = n_embed
        self.lm_head = nn.Linear(self.n_embed, config.vocab_size, bias = False)

        self.init_weights()

    def forward(self,
                batch, #(sequence ids - batchsize x seqlen, list of keywords - list of lenth batchsize, each list entry is a tensor of keywords)
                position_ids = None,
                labels = None):

        input_ids, keyword_ids = batch
        batch_size = len(keyword_ids)
        input_embedding = self.wte(input_ids) # batchsize x seqlen x embed_dim
        keyword_embedding = torch.zeros(batch_size, self.wte.embedding_dim)

        #set up keyword embeddings
        for i in range(len(keyword_ids)):
            keyword_list = keyword_ids[i]
            keyword_embeddings = self.wte(keyword_list)
            bag_of_words = torch.mean(keyword_embeddings, 0)
            keyword_embedding[i, :] = bag_of_words

        keyword_embedding = keyword_embedding.unsqueeze(1)
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
        for i, block in enumerate(self.h):
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
