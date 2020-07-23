import json
import pandas as pd
import CMUTweetTagger
#%%

with open('facebookCommenting-master/data/1595487372260.json', 'rb') as file:
    data1 = json.load(file)

with open('facebookCommenting-master/data/1595487810156.json', 'rb') as file:
    data2 = json.load(file)

example = data1[0]
example

#%%
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

#%%
output = process_comments(['diabetes', 'Diabetes'], data2)
output[1]
output2 = add_comments([output[1], output[2]], {'917455935428262': 'a test comment'})
output2
