import torch
import pickle
import json
import pandas as pd
import numpy as numpy
import transformers
from transformers import GPT2Tokenizer
from topic_link_creation import TopicLinkCreation
import bot_utils as butils
import bot_models as models
#%%

class FacebookGenerator():
    '''class to easily generate on fly/pass generations to other applications'''
    def __init__(self, model_path, model_class, keyword_list, tokenizer):
        '''model_path (str): path to folder where model is stored
           model_class (cls): the class of the model instance to be used.  Needs a load method that can accept a folder path and find the model weights in that folder.  Should also have a generate method.
           keyword_list (iterable): a list of keywords that the model was trained on'''
        self.model = model_class.load(model_path)
        self.keywords = keyword_list
        self.tokenizer = tokenizer

    def process_incoming_comment(self, comment):
        '''needs to take as input a comment and return as output the keywords in the keyword list that are in the comment'''

    def generate(prompt, max_length, top_k = None, top_p = None, num_return_sequences = 5, min_keep = 1, filter_value = -float("Inf")):
        output = self.model.generate(self.tokenizer, prompt, max_length, top_k, top_p, num_return_sequences, min_keep, filter_value)
        return output

    def process_list_of_comments(self, comment_list):
        output = [self.process_incoming_comment(comment) for comment in comment_list]
        return output




class ModelAPI():

    def __init__(self, model_file):
        self.model = ModelClass.load(model_file)
        self.path_to_clusterfile = "../data/clusters.pkl"
        self.path_to_nenp = "../data/NE+NP_v8.csv"
        self.path_to_notfound_set_topics = "../data/not_found_set-topics.xlsx"
        self.topic_link_creator = TopicLinkCreation(path_to_nenp = self.path_to_nenp, path_to_clusterfile = self.path_to_clusterfile, path_to_notfound_set_topics = self.path_to_notfound_set_topics)

    def generate_topics(text):
        '''generate topic links given a string'''
        return self.topic_link_creator.build_graph(text)

    def get_types(topic_links):
        '''show what types are present in a list of topics links'''
        type = [link[4] for link in topic_links]
        return list(set(type))

    def process_topic_links(comments, type_list = ['insurance', 'insurance+', 'insurance-', 'cost-', 'service-', 'cost']):
        '''return tweets/comments from a dict of topic links that are of a specific type'''
        output_dict = {type : [] for type in type_list}
        for key in list(comments.keys()):
            example = facebook_groups[key]
            example['id'] = key
            for topic in example['topic_links']:
                if topic[4] in type_list:
                    output_dict[topic[4]].append(example)
                else:
                    continue

        return output_dict

    def generate_(prompt):
        '''use the model to generate a comment from prompt, prompt should be of form (keyword, stem)'''
        comment_reply = model.generate(prompt)

        return comment_reply









#%%

def get_types(topic_links):
    type = [link[4] for link in topic_links]
    return list(set(type))

#%%
facebook_groups = pickle.load(open('../data/facebook_groups/topics_index_bots_fbgroups.pkl','rb'))
facebook_pages = pickle.load(open('../data/facebook_pages/topics_index_bots_fbpages.pkl', 'rb'))
#%%
len(facebook_groups)
len(facebook_pages)
example = facebook_groups[list(facebook_groups.keys())[0]]
example
#%%
example_list = []
type_list = ['insurance', 'insurance+', 'insurance-', 'cost-', 'service-', 'cost']
output_dict = {'insurance': [], 'insurance+': [], 'insurance-': [], 'cost-' : [], 'service-': [], 'cost' : [], 'service': []}
for key in list(facebook_groups.keys()):
    example = facebook_groups[key]
    for topic in example['topic_links']:
        if topic[4] in type_list:
            output_dict[topic[4]].append(example)
        else:
            continue

len(output_dict['insurance'])
len(output_dict['insurance+'])
len(output_dict['insurance-'])
len(output_dict['cost-'])
len(output_dict['service-'])
len(output_dict['cost'])
len(output_dict['service'])
#%%
output_dict['cost-'][58]
output_dict['cost'][260]
output_dict['service-'][80]

#%%
good_insurance_examples = [output_dict['insurance'][5], output_dict['insurance'][8], output_dict['insurance'][9], output_dict['insurance'][25], output_dict['insurance'][26], output_dict['insurance'][27], output_dict['insurance'][32], output_dict['insurance'][34], output_dict['insurance'][39], output_dict['insurance'][40], output_dict['cost-'][36], output_dict['cost-'][41], output_dict['cost-'][53]]

good_cost_examples = [output_dict['cost-'][3], output_dict['cost-'][36], output_dict['cost-'][41], output_dict['cost-'][48], output_dict['cost-'][53], output_dict['cost'][3], output_dict['cost'][5], output_dict['cost'][8], output_dict['cost'][68], output_dict['cost'][73], output_dict['cost'][222], output_dict['cost'][255], output_dict['service-'][931], output_dict['service-'][1065]]


#%%
model_path = '../saved_models/'
model_name1 = 'batch_051220_keyword_types_sentiment_cluster'
model_name2 = 'batch_051220_keyword_types_nosentiment_cluster'
model_name3 = 'batch_051220_keyword_types_sentiment_nocluster'

model1 = models.GPT2Model_bagofctrl.load(model_path + model_name1)
model2 = models.GPT2Model_bagofctrl.load(model_path + model_name2)
model3 = models.GPT2Model_bagofctrl.load(model_path + model_name3)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#%% Insurance replies
#get_types(good_insurance_examples[0]['topic_links'])
#good_insurance_examples[0]['tweet']fail

#%%
some_strings = ['a string', 'another string', 'a third string with a really big dick']
a_file = open('test.txt', 'w')

#%%
replace_list = ['GoodRx', '@GoodRx', '@goodrx', '#BlinkHealth', 'Blink', '@PillPack']
strp_list = ['https:']

def parameter_sweep(model, length, k_list, p_list, prompt1, prompt2, model_name):
    write_file = open('generation_output_{}.txt'.format(model_name), 'w')
    write_file.write('prompt1: {}     prompt2: {} \n'.format(prompt1, prompt2))
    outputs = {}
    for k in k_list:
        for p in p_list:
            first_sentences = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt1, 50, top_k = k, top_p = p, num_return_sequences = 10)
            second_sentences = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt2, 50, top_k = k, top_p = p, num_return_sequences = 10)
            write_file.write("k = {}  p = {}: \n---------------\n".format(k,p))
            write_file.write("First sentences:\n")
            for sentence in first_sentences[0]:
                write_file.write(sentence + '\n')

            for sentence in second_sentences[0]:
                write_file.write(sentence + '\n')

            outputs[(k,p)] = [first_sentences, second_sentences]

    write_file.close()
    return outputs

#%%
prompt1 = ['insurance-', 'insurance is']
prompt2 = ['card+', 'Use']
first_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt1, 50, top_k = 200, top_p = 0, num_return_sequences = 2)
second_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt2, 50, top_k = 200, top_p = 0, num_return_sequences = 10)
first_sentence[0]
second_sentence[0]

#%%
prompt3 = ['insurance-', 'insurance costs']
prompt4 = ['card+', 'Use']
first_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt3, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
second_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt4, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
first_sentence[0]
second_sentence[0]

#%%
prompt5 = ['insurance-', 'insurance costs']
prompt6 = ['cost+', 'Use']
first_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt5, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
second_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt6, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
first_sentence[0]
second_sentence[0]

#%%
prompt6 = ['cost-', 'insurance costs']
prompt7 = ['cost+', 'Use']
first_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt6, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
second_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt7, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
first_sentence[0]
second_sentence[0]

#%%
prompt8 = ['service-', 'insurance is']
prompt9 = ['card+', 'Use']
first_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt8, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
second_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt9, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
first_sentence[0]
second_sentence[0]

#%%
prompt10 = ['service-', 'insurance is']
prompt11 = ['service+', 'Use']
first_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt5, 50, top_k = 20, top_p = .7, num_return_sequences = )
second_sentence = butils.generate_ctrl_bagofwords(model3, tokenizer, prompt6, 50, top_k = 20, top_p = .7, num_return_sequences = 10)
first_sentence[0]
second_sentence[0]

#%%
k_list = [0,20,40,60,80,100,120,140,160,180,200]
p_list = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
length = 30
prompt1 = ['insurance-', 'insurance is']
prompt2 = ['card+', 'Use']
model = model3

output = parameter_sweep(model3, length, k_list, p_list, prompt1, prompt2, 'test')
