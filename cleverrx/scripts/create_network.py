import networkx as nx
import pandas as pd
import numpy as np

def create_network(edgelist_file)

def create_network(edgelist_file, write = True, sep = ',', filename = None, node_types = False, content_file = None):
    G = nx.DiGraph()
    edgelist = pd.read_csv(edgelist_file, sep = sep, header = None)

    for i in range(len(edgelist)):
        user_from = edgelist.iloc[i, 1]
        user_to = edgelist.iloc[i, 0]

        if G.has_edge(user_from, user_to):
            G[user_from][user_to]['weight'] += 1

        else:
            G.add_edge(user_from, user_to, weight = 1)

    if write:
         nx.drawing.nx_pydot.write_dot(G, filename)
    if node_types:
        content = pd.read_csv(content_file, sep = sep, header = None)
        for i in range(len(content)):
            G.nodes[content.iloc[i, 0]]['type'] = content.iloc[i, 1434]

    return G


def create_bipartite(content_file, attribute_number, write = True, sep = ',', filename = None, node_types = False):
    G = nx.Graph()
    content = pd.read_csv(content_file, sep = sep, header = None)
    G.add_nodes_from(range(attribute_number))
    for i in range(len(content)):
        id = content.iloc[i, 0]
        attributes = np.nonzero(content.iloc[i,1:attribute_number+1].values)[0]
        for j in attributes:
           G.add_edge(id, j)

    if write:
        nx.drawing.nx_pydot.write_dot(G, filename)

    if node_types:

        for i in range(len(content)):
            G.nodes[content.iloc[i,0]]['type'] = content.iloc[i, 1434]

    return G

def color_nodes_by_type(graph, type_string):
    groups = set(nx.get_node_Attributes(graph, type_string).values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = graph.nodes
    colors = [mapping[g.nodes[n][type_string]] for n in nodes]

    return colors
