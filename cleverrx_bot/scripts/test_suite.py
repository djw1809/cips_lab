import pytest
import pickle
import data_processing
import bot_utils as butils
import pandas as pd

##test for data processing

def test_count_hashtags():
    with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
        data = pickle.load(file)

    data = pd.DataFrame.from_dict(data, orient = 'index')
    data.index = range(len(data))
    test_data = data[0:5]
    output_dict = data_processing.count_hashtags(test_data, 'tweet')
    assert output_dict['#love.'] == 1  
