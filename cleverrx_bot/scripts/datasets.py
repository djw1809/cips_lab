import pandas as pd
import numpy as np
import pickle
#%%

##per preprocessor every dataset should ideally have a "keyword", "text" and "id" column

#%% raw_data
with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
    raw_data = pickle.load(file)

raw_data[list(raw_data.keys())[97]]
data = pd.DataFrame.from_dict(raw_data, orient = 'index')


data['id'] = data.index
data.index = range(len(data))

data.columns
len(data)

example = data.loc[97, :]
example['topic_links']

example['topic_links'][0][4].rstrip('+-')

short_data = data.loc[0:100]
short_data

#%%
npne = pd.read_csv('../data/NE+NP_v8.csv')
npne['CATEGORY'].value_counts()

# %% raw prepend, prepend words: all phrases

# %% bag of keywords with raw keywords

# %% bag of keywords,  keywords: "card", "insurance", "cost", "service" and bag of keywords, keywords: card, insurance+, insurance-, cost+, cost-, service+, service-"
def type_keywords(input_data, id_field, text_field, topic_link_field, sentiment = True):
    output_data = pd.DataFrame(columns = ['id', 'text', 'keywords'])

    for entry in input_data.index:
        output_data.loc[len(output_data) + 1] = np.nan #preallocate memory
        id = input_data.loc[entry, id_field]
        text = input_data.loc[entry, text_field]
        topic_links = input_data.loc[entry, 'topic_links']
        if sentiment:
            type_keywords = [j[4] for j in topic_links if len(j[4]) > 0]
            type_keywords = list(set(type_keywords)) #no duplicates
        else:
            type_keywords = [j[4].rstrip('+-') for j in topic_links if len(j[4]) > 0] #only get keywords that have a type and strip off the sentiment
            type_keywords = list(set(type_keywords)) #no duplicates

        output_data.loc[output_data.index.max()] = {'id': id, 'text':text, 'keywords':type_keywords}
        output_data.index = range(len(output_data))
    return output_data

test_output_sentiment = type_keywords(short_data, 'id', 'tweet', 'topic_links')
test_output_nosentiment = type_keywords(short_data, 'id', 'tweet', 'topic_links', sentiment = False)

test_output_sentiment
test_output_nosentiment

# # %%  bag of keywords, keywords: card, insurance+, insurance-, cost+, cost-, service+, service-, all cluster names
# def type_cluster_keywords(input_data, id_field, text_field, topic_link_field, sentiment = True):
#     output_data = pd.DataFrame(columns ['id', 'text', 'keywords'])
#
#     for entry in input_data.index:
#         output_data.loc[output_data.index.max() + 1] = np.nan
#         id = input_data.loc[entry, id_field]
#         text = input_data.loc[entry, text_field]
#         topic_links = input_data.loc[entry, 'topic_links']
#
#         if sentiment:
#             type_keywords = [j[4] for j in topic_links if len(j[4]) > 0]
#             cluster_keywords = [j[0] for j in topic_links if len(j[0]) > 0]
#             type_keywords = list(set(type_keywords))
#             cluster_keywords = list(set(cluster_keywords))
#             keywords = type_keywords + cluster_keywords
#
#         else:
#             type_keywords = [j[4].rstrip('+-') for j in topic_links if len(j[4]) > 0]
#             cluster_keywords = [j[0] for j in topic_links if len(j[0]) > 0]
#             type_keywords = list(set(type_keywords)) #no duplicates
#             cluster_keywords = list(set(cluster_keywords))
#             keywords = type_keywords + cluster_keywords
#
#         output_data.loc[output_data.index.max()] = {'id': id, 'text':text, 'keywords':keywords}
#
#     return output_data
#
#
# # %% raw prepending with all the same as above
# def type_prepend(input_data, id_field, text_field, topic_link_field, sentiment = True):
#     output_data = pd.DataFrame(columns = ['id', 'prepended_text'])
#
#     for entry in input_data.index:
#         output_data.loc[output_data.index.max() + 1] = np.nan #preallocate memory
#         id = input_data.loc[entry, id_field]
#         text = input_data.loc[entry, text_field]
#         topic_links = input_data.loc[entry, 'topic_links']
#         if sentiment:
#             type_keywords = [j[4] for j in topic_links if len(j[4]) > 0]
#             type_keywords = list(set(type_keywords)) #no duplicates
#         else:
#             type_keywords = [j[4].rstrip('+-') for j in topic_links if len(j[4]) > 0] #only get keywords that have a type and strip off the sentiment
#             type_keywords = list(set(type_keywords)) #no duplicates
#
#         for keyword in type_keywords:
#             text = keyword + ' ' + text


#%%

def prepare_keyword_dataset(input_data, id_field, text_field, topic_link_field, sentiment = False, cluster = False):
    '''prepares dataframe with different ways of handling keywords.
       input: a dataframe with a field for text_ids, texts, and a topic link field. A topic link is a list of tuples of the form (cluster, phrase, phrase_synonym, disease, type).  Each text has a topic link for each phrase present in text. Keywords for a text always at least include the type.
       sentiment: include sentiment symbol in type keywords
       cluster: include cluster names in keywords
       output:
       if prepend: a dataframe with id, and prepend_text fields.  prepend_text is original text with keywords prepended.
       otherwise: a dataframe with id, text and keyword fields'''

    output_data = pd.DataFrame(columns = ['id', 'text', 'keywords'])

    for entry in input_data.index:
        output_data.loc[len(output_data) + 1] = np.nan
        id = input_data.loc[entry, id_field]
        text = input_data.loc[entry, text_field]
        topic_links = input_data.loc[entry, 'topic_links']

        if sentiment:
            type_keywords = [j[4] for j in topic_links if len(j[4]) > 0]
        else:
            type_keywords = [j[4].rstrip('+-') for j in topic_links if len(j[4]) > 0]

        if cluster:
            cluster_keywords = [j[0] for j in topic_links if len(j[0]) > 0]
            type_keywords = list(set(type_keywords))
            cluster_keywords = list(set(cluster_keywords))
            keywords = type_keywords + cluster_keywords
        else:
            type_keywords = list(set(type_keywords))
            keywords = type_keywords


        output_data.loc[output_data.index.max()] = {'id': id, 'text':text, 'keywords':keywords}

    output_data.index = range(len(output_data))

    return output_data

#keyword tests
test1 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links') #just types no sentiment
test2 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links', sentiment = True) #types with sentiment
test3 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links', cluster = True) #types and clusters no sentiment
test4 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links', sentiment = True, cluster = True) #types and clusters sentiment

test1
test2
test3
test4

#prepend tests
test5 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links', prepend = True)
test6 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links', prepend = True, sentiment = True)
test7 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links', prepend = True, cluster = True)
test8 = prepare_keyword_dataset(short_data, 'id', 'tweet', 'topic_links', prepend = True, sentiment = True, cluster = True)

test5
test6
test7
test8




# %% bag of keywords with all of above + prepending of phrase synonym
