import json
import pickle
import pandas as pd
import CMUTweetTagger
import numpy as numpy
from topic_link_creation import TopicLinkCreation
import bot_utils as butils
import bot_models as models
import facebook_data_processing as fb
#%%

#%%get data
data_path = 'facebookCommenting-master/data/1595553517145.json'
keep_list_path = 'facebookCommenting-master/data/072320_evening_keep_list.pkl'

# with open(data_path, 'rb') as file:
#     data = json.load(file)


#%%view comments/construct keep list
with open(keep_list_path, 'rb') as file:
    keep_list = pickle.load(file)

#%%automatically select things to keep using targets

##%%onstruct comment dict
comment_dict = fb.create_comment_dict(keep_list)
comment_dict
##generate comment salvo and save it
salvo = fb.add_comments(keep_list, comment_dict)


with open('evening_072320_salvo.json', 'w') as file:
    json.dump(salvo, file)
