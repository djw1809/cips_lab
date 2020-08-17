import json
import pandas as pd
import CMUTweetTagger
import numpy as numpy
import transformers
from transformers import GPT2Tokenizer
from topic_link_creation import TopicLinkCreation
import bot_utils as butils
import bot_models as models
#%%

#%%
# output = process_comments(['diabetes', 'Diabetes'], data2)
# output[1]
# output2 = add_comments([output[1], output[2]], {'917455935428262': 'a test comment'})
# output2
# #%%
# diabetes_entities = pd.read_csv('../results/diabetes_dataframe_short.csv')
# diabetes_entities
# sorted = diabetes_entities.sort_values(by = ['count'], ascending = False)
# sorted.index = range(len(sorted))
# sorted[0:200].to_csv('../results/diabetes_dataframe_short_top_200.csv')
# target_list = ['Diabetes', 'diabetes', 'lifestyle', 'diabetic', 'insulin', 'treatment', 'insurance', 'problems', 'suffering', 'chronic']
#%%


#functions for general data processing

def produce_entity_list(data):
    output_dict = {}
    for i in data.index:
        tweet = [data.loc[i, 'content']]
        ents = CMUTweetTagger.runtagger_parse(tweet)[0];
        for ent in ents:
            if ent[0] in output_dict.keys():
                output_dict[ent[0]]['count']+=1
            else:
                if ent[1] in ['N', '^', 'S', 'Z', 'M', 'A']:
                    enty = ent[0]
                    pos = ent[1]
                    output_dict[enty] = {'pos': pos, 'count': 1}

    output_df = pd.DataFrame.from_dict(output_dict, orient = 'index')
    output_df['entity'] =output_df.index
    output_df.index = range(len(output_df))
    return output_df, output_dict

def count_hashtags(data, content_field):
    output_dict = {} #hashtag/accountname:occurence count
    for i in data.index:
        tweet = data.loc[i, content_field]
        hashtags = [tag for tag in tweet.split() if (tag.startswith('#') or tag.startswith('@'))]
        for tag in hashtags:
            if tag in output_dict.keys():
                output_dict[tag] += 1
            else:
                output_dict[tag] = 1

    return output_dict


#functions for processing lists of posts/adding comments
def process_comments(target_list, post_list):
    hits = []
    for comment in post_list:
        hit_ents = []
        content = [comment['content']]
        ents = CMUTweetTagger.runtagger_parse(content)[0];
        for ent in ents:
            enty = ent[0]
            if enty in target_list:
                hit_ents.append(enty)

        if len(hit_ents) > 0:
            comment['hit_ents'] = hit_ents
            hits.append(comment)

    return hits

def add_comments(post_list, comment_dict):
    for key in comment_dict.keys():
        for post in post_list:
            if post['postid'] == key:
                post['comment'] = comment_dict[key]

    return post_list

def check_posts(post_list, save = False, save_location = 'facebookCommenting-master/data/', title = None):
    keep_list = []
    for post in post_list:
        print(post['content'])
        keep = input()
        if keep == '1':
            keep_list.append(post)
        else:
            pass

    if save:
        with open(save_location + title, 'wb') as file:
            json.dump(keep_list, file)
    return keep_list

def create_comment_dict(post_list, save = False, save_location = 'facebookCommenting-master/data/', title = None):
    comment_dict = {}
    for post in post_list:
        print(post['content'])
        id = post['postid']
        comment = input("input a comment you would like to make on this post")
        comment_dict[id] = comment

    if save:
        with open(save_location +title, 'wb') as file:
            json.dump(comment_dict, file)

    return comment_dict



#maybe will need this much later
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
