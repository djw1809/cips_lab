import numpy as np
import networkx as nx
import pandas as pd
import pickle
#import cleverrx_analysis_utils as c
import re 
import nltk 
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.util import ngrams 


def split_string_into_clean_sentences(string):
    '''Splits a string into a list of sentences.  removes special characters from sentences and lowercases everything.''' 
    sentences = sent_tokenize(string)
    for i in range(len(sentences)):
        sentence = sentences[i] 
        sentence = sentence.lower()
        sentence = ''.join(s for s in sentence if ord(s)>31 and ord(s)<126)
        sentence = re.sub(r"([,()*&^%$\n])", r"", sentence) 
        sentence = sentence.rstrip().lstrip().lower() 
        sentences[i] = sentence 
    return sentences 
        

def check_sentence_phrase(sentence, phrase):
    '''checks if a phrase is present in a certain sentence - meant to check if a context for a whole comment is present in one of its sentences'''
    phrase = phrase.lstrip().rstrip() 
    phrase_words = word_tokenize(phrase)
    context_length = len(phrase_words)
    
    ##compute all grams of length equal to context_length 
    sentence_words = word_tokenize(sentence)
    grams = ngrams(sentence_words, context_length)
    
    if tuple(phrase_words) in grams:
        return True 
    
    else:
        return False 
    
def check_sentence_phrase_list(sentence, phrase_list, synonyms = []): 
    '''checks if any phrase from phrase list is present in sentence'''
    phrase_list_tuples = [tuple(word_tokenize(i)) for i in phrase_list]
    maximum_phrase_length = max([len(i) for i in phrase_list_tuples])
    sentence_words = word_tokenize(sentence)
    sentence_phrases = []
    for i in range(1, maximum_phrase_length+1): #check for all possible phrase lengths 
        #for each possible phrase length compute the ngrams 
        grams = ngrams(sentence_words, i)
        for gram in grams:
            #check if each ngram is in the phrase list 
            if gram in phrase_list_tuples:
            #if it is make the tuple of the ngram into a string and add it to the sentence phrases optionally adding a synonum from a given dict instead      
                phrase = gram[0] 
                for i in range(1, len(gram)):
                    phrase = phrase + ' ' + gram[i]
                
                if len(synonyms) != 0: 
                    if phrase in synonyms.keys():
                        sentence_phrases.append(synonyms[phrase])
                    else: 
                        sentence_phrases.append(phrase)
                    
                else:
                    sentence_phrases.append(phrase) 
                
    return list(set(sentence_phrases)) 

def build_cooccurence_dict(comment_list, phrase_list, synonyms = [], graph = False): 
    edge_dict = {} 
    sentence_list = [] 
    for comment in comment_list: 
        comment = str(comment) #incase of numpy string 
        sentences = split_string_into_clean_sentences(comment) #look at comments sentence wise 
        for sentence in sentences: 
            sentence_list.append(sentence) #keep track of what were actually iterating over 
            
            phrases = check_sentence_phrase_list(sentence, phrase_list, synonyms) #compute phrases that cooccur in the sentence 
            if len(phrases) > 1: #if there are occuring phrases add the edge to the edge dict, if it already exists update weight
                for i in range(len(phrases)):
                    for j in [j for j in range(len(phrases)) if j > i]:
                        if (phrases[i], phrases[j]) in edge_dict.keys():
                            edge_dict[(phrases[i], phrases[j])] += 1 
                        else: 
                            edge_dict[(phrases[i], phrases[j])] = 1 
    
    if graph: #optionally build networkx graph object from edge_dict 
        
        ebunches = [(key[0], key[1], {'weight':edge_dict[key]}) for key in edge_dict.keys()]
        G = nx.Graph() 
        G.add_edges_from(ebunches) 
        return sentence_list, edge_dict, G 

    else:
        
        return sentence_list, edge_dict 
        
                    
    
        
        
        
        


#############full data################3
#%%
with open('../data/cleaned_user_features.pkl', 'rb') as f:
    full_data = pickle.load(f)
#flatten to by text 
flattened_full_data = []
for user in full_data.keys():
    for comment in full_data[user]:
        flattened_full_data.append(comment)

#add a text id 
for i in range(len(flattened_full_data)):
    flattened_full_data[i]['text_id'] = i

#only keep comments with more then one context 
texts_with_multiple_contexts = []
for text in flattened_full_data:
    if len(text['contexts']) >= 2:
        texts_with_multiple_contexts.append(text)

synonyms = pd.read_csv('../coded.csv', sep = ',')
synonyms = synonyms.dropna()
synonym_dict = synonyms.set_index('Phrase').T.to_dict('records')[0] 

#replace using aadhaven's synonym list 
for i in range(len(texts_with_multiple_contexts)): 
    text = texts_with_multiple_contexts[i]
    for context in text['contexts'] :
        if context['phrase'] in synonym_dict.keys(): 
            context['phrase'] = synonym_dict[context['phrase']]

###########################################
        




#%% 
sentence_data = pd.DataFrame(columns = ['sentence', 'contexts', 'user'])

#### build sentence/context list 
for i in range(len(texts_with_multiple_contexts)):
    text_object = texts_with_multiple_contexts[i]
    text = text_object['text']
    sentences = split_and_clean_string(text) 
    user = text_object['user']
    for i in range(len(sentences)): 
        sentences[i] = sentences[i].rstrip().lstrip() 
        
    
    for sentence in sentences: 
        sentence_contexts = [] 
        for context in text_object['contexts']: 
            if check_sentence_context(sentence, context['phrase']):
                sentence_contexts.append(context)
            else:
                pass 
        
        row = pd.DataFrame([[sentence, sentence_contexts, user]], columns = ['sentence', 'contexts', 'user'])
        sentence_data = sentence_data.append(row)

sentence_data.index = range(len(sentence_data))        
#%% build cooccurence list
cooccurence = pd.DataFrame(columns = ['context1', 'context2'])

for k in range(len(sentence_data)):
    sentence_contexts = sentence_data.loc[k, 'contexts']
    for i in range(len(sentence_contexts)):
        for j in [j for j in range(len(sentence_contexts)) if j > i]:
            row = pd.DataFrame([[sentence_contexts[i], sentence_contexts[j]]], columns = ['context1', 'context2'])
            cooccurence = cooccurence.append(row) 
            
            
#%%



#%%

# test = False
# ######context-context graph#######
# with open('../data/old_fb/fb_graph.pkl', 'rb') as f:
#     file = pickle.load(f)
# fb_graph = pd.DataFrame(file, columns = ['user', 'context', 'experience', 'text', 'other'])
# fb_graph = fb_graph.drop(columns = ['other'])
#
# cleaner_graph = fb_graph.loc[fb_graph.context != ('','')]
# cleaner_graph.index = range(len(cleaner_graph))
#
# context_graph = pd.DataFrame()
# context_graph['user'] = cleaner_graph['user']
# context_graph['context'] = [cleaner_graph.loc[i, 'context'][0] for i in range(len(cleaner_graph))]
# context_graph['text'] = cleaner_graph['text']
# context_graph = context_graph[context_graph.context != 'not_found']
# context_graph.index = range(len(context_graph))
#
# df = context_graph.drop(columns = ['text'])
# user_context_bipartite = c.create_bipartite(df, 'user', 'context')
# context_context_graph = c.create_two_step(user_context_bipartite, 'context', 'user', 'type')
# largest_cc = max(nx.connected_components(context_context_graph), key=len)
# cleaned_context_context = context_context_graph.subgraph(largest_cc).copy()





# def create_flattened_context(flattened_full_data):
#
#
#     context_flattened_data = pd.DataFrame(columns = ['user', 'text', 'text_id', 'phrase', 'synonym', 'type', 'extra_info'])
#
#     for text in flattened_full_data:
#         for context in text['contexts']:
#             append = pd.DataFrame([[text['user'], text['text'], text['text_id'], context['phrase'], context['additional_word'], context['type'], context['extra_info']]], columns = ['user', 'text', 'text_id', 'phrase', 'synonym', 'type', 'extra_info'] )
#             context_flattened_data = context_flattened_data.append(append)
#
#     return context_flattened_data
#
# context_flattened_data = create_flattened_context(flattened_full_data)




#######testing###################333
# if __name__ == '__main__':
#     if run:
#         df = context_graph.drop(columns = ['text'])
#         user_context_bipartite = c.create_bipartite(df, 'user', 'context')
#         context_context_graph = c.create_two_step(user_context_bipartite, 'context', 'user', 'type')
#         really_bipartite = [i for i in user_context_bipartite.edges if user_context_bipartite.nodes[i[0]]['type'] == user_context_bipartite.nodes[i[1]]['type']]
#
#     if test:
#         test_graph_df = pd.DataFrame()
#         test_graph_df['group_1'] = [0, 0, 1, 2, 2]
#         test_graph_df['group_2'] = [4,5,5,4,6]
#         test_bipartite = c.create_bipartite(graph_df, 'group_1', 'group_2')
#         test_context = c.create_two_step(test_bipartite, 'group_2', 'group_1', 'type')
