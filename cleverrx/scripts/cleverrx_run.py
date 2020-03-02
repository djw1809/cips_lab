import numpy as np
import networkx as nx
import pandas as pd
import pickle
import cleverrx_analysis_utils as c



test = False
### get data
with open('../data/fb_graph.pkl', 'rb') as f:
    file = pickle.load(f)
fb_graph = pd.DataFrame(file, columns = ['user', 'context', 'experience', 'text', 'other'])
fb_graph = fb_graph.drop(columns = ['other'])

cleaner_graph = fb_graph.loc[fb_graph.context != ('','')]
cleaner_graph.index = range(len(cleaner_graph))

context_graph = pd.DataFrame()
context_graph['user'] = cleaner_graph['user']
context_graph['context'] = [cleaner_graph.loc[i, 'context'][0] for i in range(len(cleaner_graph))]
context_graph['text'] = cleaner_graph['text']
context_graph = context_graph[context_graph.context != 'not_found']
context_graph.index = range(len(context_graph))

df = context_graph.drop(columns = ['text'])
user_context_bipartite = c.create_bipartite(df, 'user', 'context')
context_context_graph = c.create_two_step(user_context_bipartite, 'context', 'user', 'type')


if __name__ == '__main__':
    if run:
        df = context_graph.drop(columns = ['text'])
        user_context_bipartite = c.create_bipartite(df, 'user', 'context')
        context_context_graph = c.create_two_step(user_context_bipartite, 'context', 'user', 'type')
        really_bipartite = [i for i in user_context_bipartite.edges if user_context_bipartite.nodes[i[0]]['type'] == user_context_bipartite.nodes[i[1]]['type']]

    if test:
        test_graph_df = pd.DataFrame()
        test_graph_df['group_1'] = [0, 0, 1, 2, 2]
        test_graph_df['group_2'] = [4,5,5,4,6]
        test_bipartite = c.create_bipartite(graph_df, 'group_1', 'group_2')
        test_context = c.create_two_step(test_bipartite, 'group_2', 'group_1', 'type')
