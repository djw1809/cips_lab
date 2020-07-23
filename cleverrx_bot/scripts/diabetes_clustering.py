import pandas as pd
import numpy as np
import CMUTweetTagger
import json

#%%
groups_raw = pd.read_json('../data/facebookgroups.json')
groups_topic_index = pd.read_json('../data/facebook_groups/topics_index_bots_fbgroups.json', orient = 'index')
pages_raw = pd.read_json('../data/facebookpages.json')
pages_topic_index = pd.read_json('../data/facebook_pages/topics_index_bots_fbpages.json', orient = 'index')
#%%

example =groups_raw.loc[0, 'content']
output = CMUTweetTagger.runtagger_parse([example])
ent = output[0][1]
ent
#%%
raw = groups_raw.append(pages_raw)
groups_raw
pages_raw
raw
raw.index = range(len(raw))


#%%

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

    output_df = pd.DataFrame.from_dict(short_dict, orient = 'index')
    output_df['entity'] =df.index
    output_df.index = range(len(df))
    return output_df, output_dict

#%%
if __name__ == '__main__':
    groups_raw = pd.read_json('../data/facebookgroups.json')
    pages_raw = pd.read_json('../data/facebookpages.json')
    raw = groups_raw.append(pages_raw)
    raw.index = range(len(raw))
    output = produce_entity_list(raw)
    output[0].to_csv('diabetes_dataframe.csv')
