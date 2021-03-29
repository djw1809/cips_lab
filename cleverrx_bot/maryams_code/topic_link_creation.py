import os
import time
import csv
import pickle
import pandas as pd
import re
import numpy as np
import CMUTweetTagger
from tqdm import tqdm
from collections import defaultdict
import json

class TopicLinkCreation:

    def __init__(self, tweet = 'This is a sample input about health', path_to_nenp = "../data/topic_link_data/NE+NP_v8.csv", path_to_clusterfile = "../data/topic_link_data/clusters.pkl",
                path_to_notfound_set_topics = "../data/topic_link_data/not_found_set-topics.xlsx"):

        self.tweet = tweet
        self.path_to_nenp = path_to_nenp
        self.path_to_clusterfile = path_to_clusterfile
        self.path_to_notfound_set_topics = path_to_notfound_set_topics

    def clean_tweets(self, tweet):

        print("--- Cleaning Tweets ---")
        tweet = (tweet.encode("ascii", errors="ignore").decode()).lower().strip()
        tweet = tweet.replace("\n", "")
        tweet = tweet.replace("\r", "")
        tweet_text = tweet.strip()
        print("--- END Cleaning Tweets ---")
        return tweet_text

    def get_ctxt_exp_list(self):

        # the input to this function (path_to_nenp) is the NE+NP_v8.csv or equal to this file
        exp_context_list_dict = dict()
        meds_disease_dict = dict()
        phrase_synonym_dict = dict()
        df = pd.read_csv(self.path_to_nenp , header = 0)
        # print(df.head())
        # print((np.unique(df[["CATEGORY"]].values)))
        for (key, val, dis, syn) in df[['PHRASE', 'CATEGORY', 'DISEASE', 'SYNONYMS-OF']].values:
            key = (key.lower()).replace("#", "").strip()
            val = (val.lower()).strip()
            exp_context_list_dict[key] = val

            if val == 'medication' and dis == dis:
                meds_disease_dict[key] = (dis.lower()).strip()
            elif val == 'medication' and dis != dis:
                print("Find Dis for: ", key)
            if syn == syn:
                phrase_synonym_dict[key] =  syn
        return exp_context_list_dict, meds_disease_dict, phrase_synonym_dict

    def get_cluster_data(self):

        # input of this function is the path (string) to the clusterfile. The defult value will be used
        # if no path is provided
        with open(self.path_to_clusterfile, "rb") as file:
            data = pickle.load(file)

            # reverse dict <>
            phrase_to_cluster = dict()
            for key, val in data.items():
                for v in val:
                    phrase_to_cluster[v.lower().strip()] = key.lower().strip()

        df = pd.read_excel(self.path_to_notfound_set_topics, header = None)
        temp = df.values
        for (i,j) in temp:
            phrase_to_cluster[i.lower().strip()] = j.lower().strip()

        res = phrase_to_cluster.values()
        print(set(res))
        print(len(set(res)))
        return phrase_to_cluster

    def new_ne_extraction(self, ne_tags):

        temp = []
        for tag in ne_tags:
            temp.append(tag[0])
        return temp

    def build_graph(self, tweet):

        # input of this function is a text in the str format, "this is an sample input"
        exp_context_list_dict, meds_disease_dict, phrase_synonym_dict = self.get_ctxt_exp_list()
        cluster_dict = self.get_cluster_data()
        # Getting Np/NE
        cleaned_tweet = self.clean_tweets(tweet)
        entity_results = CMUTweetTagger.runtagger_parse([cleaned_tweet]) # the input should be a list of texts
    #     print("3- entity_results:" , entity_results)
        print("--- End Tagging Tweets ---")
        print("Tagged Ents: ", len(entity_results))


        # For each tweet
        for i in tqdm(range(len(entity_results))):
            phrases_list = set(self.new_ne_extraction(entity_results[i]))
            type_list = []
            b_syn_list = []
            topic_list = []
            disease_list = []
            topic_links = []
            if len(phrases_list) > 0:
                for ent in phrases_list:
                    ent = (ent.replace("#", "")).lower().strip()
                    ent = ''.join(e for e in ent if e.isalnum())
                    # if len(ent)>0:
                    #     if ent[0].isalpha() == False:
                    #         ent = ent[1:]
                    # Types
                    type_word = None
                    if ent in exp_context_list_dict.keys():
                        type_word = exp_context_list_dict.get(ent)
                        type_list.append(type_word)
                    else:
                        type_list.append("")
                    # B-syn list
                    if ent in phrase_synonym_dict.keys():
                        synonym = phrase_synonym_dict.get(ent)
                        b_syn_list.append(synonym)
                    else:
                        b_syn_list.append("")
                    # if "blue" in ent.lower().strip():
                    #         print("Hit   ", ent.lower().strip())
                    if ent.lower().strip() in cluster_dict.keys():
                        topic_list.append(cluster_dict.get(ent.lower().strip()))

                    else:
                        topic_list.append("")

                    if type_word == "medication":
                        if ent in meds_disease_dict.keys():
                            disease_list.append(meds_disease_dict.get(ent))
                        else:
                            disease_list.append("")
                    else:
                        disease_list.append("")

            topics = list(set(topic_list))

            if "" in topics:
                topics.remove("")

            for (phrase,typ,b_syn,topic,m) in zip(phrases_list,type_list, b_syn_list, topic_list, disease_list):
                if typ != "" or topic!= "":
                    topic_links.append((topic,phrase,b_syn,m,typ))

            if topic_links==[]:
                # print("Hit")
                continue
#                 print(phrase,",",typ,",",b_syn,",",topic,",",m)

            tweet_dict = {}
            tweet_dict["topics"] = topics
            tweet_dict["topic_links"] = topic_links
            tweet_dict["tweet"] = tweet

#             print(tweet_dict)
#             print("--------------------------------------------")

        return tweet_dict
