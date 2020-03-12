import numpy as np
#import pandas as pd
import networkx as nx
#import pickle
import heapq
#from networkx.algorithms.operators.binary import disjoint_union, union
from sklearn.cluster import KMeans
from itertools import count
import matplotlib.pyplot as plt
#import heapq
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib
import ast
#from node2vec import Node2Vec 
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.util import ngrams 
import re

#####Graph Generators ########


def create_bipartite(graph_df, group_1, group_2, write = False, filename = None):
    G = nx.Graph()

    group_1_nodes = graph_df.loc[:, group_1].unique()
    group_2_nodes = graph_df.loc[:, group_2].unique()

    G.add_nodes_from(group_1_nodes, type = group_1)
    G.add_nodes_from(group_2_nodes, type = group_2)

    for i in range(len(graph_df)):
        node1 = graph_df.loc[i, group_1]
        node2 = graph_df.loc[i, group_2]

        if G.has_edge(node1, node2):
            G[node1][node2]['weight'] += 1

        else:
            G.add_edge(node1, node2, weight = 1)

    if write:
        nx.drawing.nx_pydot.write_dot(G, filename)

    return G

def create_two_step(bipartite_graph, graph_group, bridge_group, type_string, write = False, filename = None):

    G_bi = bipartite_graph
    graph_group_nodes = [i for i in G_bi if G_bi.nodes[i][type_string] == graph_group]
    bridge_group_nodes = [i for i in G_bi if G_bi.nodes[i][type_string] == bridge_group]


    G = nx.Graph()


    G.add_nodes_from(graph_group_nodes, type_string = graph_group)

    for i in bridge_group_nodes:
        for j in G_bi[i]:
            for k in [m for m in G_bi[i] if m > j]:

                if G.has_edge(j,k):
                    G[j][k]['weight'] += 1
                else:
                    G.add_edge(j,k, weight = 1)

    if write:
        nx.drawing.nx_pydot.write_dot(G, filename)

    return G

#####algorithms#######

def compute_matricies(G, deg_out = False):
    '''computes Laplacian and normalized Laplacian of a graph'''

    adj = nx.convert_matrix.to_numpy_matrix(G)
    if deg_out:
        deg = [np.sum(adj[i, :]) for i in range(len(G))]

    else:
        deg = [np.sum(adj[:, i]) for i in range(len(G))]

    D = np.diag(deg)
    D_half = np.diag(np.divide(1,np.sqrt(deg), out = np.zeros_like(deg), where=np.array(deg)!=0))

    L = D - adj
    norm_L = D_half * L * D_half

    norm_A = D_half * adj

    return L, norm_L, norm_A, D_half, deg


def n_cut_classification(G, matrix_choice = 0, num_v_start = 1, num_v_stop = 2, v_list = None, plot_latent = False, plot_graph = False, show_labels = False, return_labels = True, clusters = 2, deg_out = False):
    '''computes and visualizes a clustering of the graph G'''

    ##relabel G to handle graphs with non-sequential labels
    #G_labels = nx.relabel.convert_node_labels_to_integers(G)

    #use the laplacian
    if matrix_choice == 0:
        L = compute_matricies(G, deg_out)[0]

    #use the normalized laplacian (N-cut)
    elif matrix_choice == 1:
        L = compute_matricies(G, deg_out)[1]

    elif matrix_choice == 2:
        L = compute_matricies(G, deg_out)[2]


    #calculate eigenvectors/values
    w,v = np.linalg.eig(L)
    w = np.real(w)
    v = np.real(v)

    #if a list of indicies is not passed, sort the eigenvalues in decreasing order and pick the eigenvectors starting with and including num_v_start and ending with and NOT including num_v_end
    if v_list == None:
        lambda_indicies = heapq.nsmallest(num_v_stop+1, range(len(w)), key = w.__getitem__)[num_v_start:num_v_stop]
    #otherwise use a list of indicies passed and grab from the list v - NOTE: v is NOT in general sorted! (no way to know which index corresponds to largest/smallest eigenvalue)
    else:
        lambda_indicies = v_list
    latent_rep = np.column_stack([v[:, i] for i in lambda_indicies])

    ###k-means cluster the coordinates of v
    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(latent_rep)
    labels = kmeans.labels_

    if return_labels:
        for i in range(len(labels)):
            G.nodes[list(G.nodes)[i]]['label'] = labels[i]

    if plot_latent:

        if len(lambda_indicies) == 1:
            plt.scatter(np.array(latent_rep[:,0]), np.zeros(len(G)), c = labels, cmap = plt.cm.jet)



        elif len(lambda_indicies) == 2:
            plt.scatter(np.array(latent_rep[:,0]), np.array(latent_rep[:,1]), c = np.array(labels).reshape(len(G), 1), cmap = plt.cm.jet)

        # elif len(lambda_indicies) == 3:
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #
        #     ax.scatter(np.array(latent_rep[:, 0]), np.array(latent_rep[:, 1]), np.array(latent_rep[: ,2]), c = np.array(labels).reshape(len(G_labels), 1), cmap = plt.cm.jet)

        plt.show()
        plt.clf()

    if plot_graph:

        pos = nx.drawing.layout.spring_layout(G)
        edges = nx.draw_networkx_edges(G, pos, alpha = 0.2)
        nodes = nx.draw_networkx_nodes(G, pos, nodelist = G.nodes, node_color = labels, node_size = 200, cmap = plt.cm.jet, with_labels = True)

        plt.show()


    return G, w, v, latent_rep, labels

def recursive_automatic_ncut(G, steps, matrix_choice, num_v_start = 1, num_v_stop = 2, manual_keep = False, automatic_keep = False, keep_threshhold = 0, vlist = None, plot_graph = True, draw = False):

    total_ncut = 0
    clusterings = []
    normalized_ncut_values = []

    for i in range(len(G)):
        G.nodes[list(G.nodes)[i]]['label'] = 0

    clusterings.append(G.copy())



    for i in range(steps):
        #G_step = G.copy()
        n_cut_step = 0
        clusters = list(set(nx.get_node_attributes(G, 'label').values()))
        keep_clusters = [] 
        if manual_keep and i != 0:
            keep_clusters = ast.literal_eval(input("enter a list of clusters you'd like to keep"))
            run_clusters = [i for i in clusters if i not in keep_clusters ]

        elif automatic_keep:
            keep_clusters = []
            for j in clusters:
                cluster = [node for node in G.nodes if clusterings[i].nodes[node]['label'] == j]
                if len(cluster) <= keep_threshhold:
                    keep_clusters.append(j)


            run_clusters = [i for i in clusters if i not in keep_clusters]
            
        else:
            run_clusters = clusters

        for j in run_clusters:
            G_cluster = G.subgraph([node for node in G.nodes if G.nodes[node]['label'] == j]).copy()
            #if len(G_cluster) == 1:
                #print('there is a 1 cluster likely failure')
            clustered_G = n_cut_classification(G_cluster, matrix_choice = 1)[0]
            label_0 = [node for node in clustered_G.nodes if clustered_G.nodes[node]['label'] == 0]
            label_1 = [node for node in clustered_G.nodes if clustered_G.nodes[node]['label'] == 1]
            #n_cut_step += nx.algorithms.cuts.normalized_cut_size(G, label_0, label_1)

            assigned_label_1 = max(list(set(nx.get_node_attributes(G, 'label').values()))) +1
            assigned_label_2 = max(list(set(nx.get_node_attributes(G, 'label').values()))) + 2
            for node in label_0:
                G.nodes[node]['label'] = assigned_label_1 #(i+1) * j #j*2^(i-1) #max(list(set(nx.get_node_attributes(G_step, 'label').values()))) + 1     #

            for node in label_1:
                G.nodes[node]['label'] = assigned_label_2  #(i+1) * j + 1 #j*2^(i-1) + 1 #max(list(set(nx.get_node_attributes(G_step, 'label').values()))) + 1    #

        total_ncut += n_cut_step
        normalized_n_cut = np.divide(total_ncut, 2*len(run_clusters) + len(keep_clusters))
        normalized_ncut_values.append(normalized_n_cut)
        clusterings.append(G.copy())

        if draw:
            pos = nx.drawing.layout.spring_layout(G)
            edges = nx.draw_networkx_edges(G, pos, alpha = 0.2)
            norm = matplotlib.colors.Normalize(vmin = 0, vmax = 4*(2**i))
            cmap = plt.cm.jet
            colors = []
            patches = []
            node_labels = nx.get_node_attributes(G, 'label')
            for k in sorted(node_labels):
                colors.append(node_labels[k])


            for k in set(colors):
                patches.append(mpatches.Patch(color = cmap(norm(k)), label = str(k)))

            colors = np.array(colors)
            nodes = nx.draw_networkx_nodes(G, pos, nodelist = G.nodes, node_color = colors, node_size = 50, cmap = cmap, with_labels = False, vmin = 0, vmax =4*(2**i))
            plt.legend(handles = patches)
            plt.show()
        
        cluster_dicts = [] 
        for graph in clusterings: 
            #%%
            cluster_dict = {} 
            node_keys = list(graph.nodes)
            for key in node_keys: 
                if graph.nodes[key]['label'] in cluster_dict.keys():
                    cluster_dict[graph.nodes[key]['label']].append(key) 
                else: 
                    cluster_dict[graph.nodes[key]['label']] = [key] 
#%%            
            cluster_dicts.append(cluster_dict)


    return clusterings, cluster_dicts, normalized_ncut_values


def node2vec_classification(G, clusters, dim = 128, walk_length = 80, num_walks = 10, return_ =1, inout = 1):
    
    node2vec = Node2Vec(G, dimensions = dim, walk_length = walk_length, num_walks = num_walks, p = return_, q = inout)
    model = node2vec.fit(window = 10, min_count = 1, batch_words = 4)
    word_vector_matrix = np.vstack([model.wv[node] for node in list(G)])
    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(word_vector_matrix)
    labels = kmeans.labels_
    node_labels = zip(list(G), labels)
    clusters = {} 
    for pair in node_labels: 
        
        if pair[1] in clusters.keys():
            clusters[pair[1]].append(pair[0])
        
        else:
            clusters[pair[1]] = [pair[0]] 
    
    return clusters, word_vector_matrix 
    


### Create phrase cooccurence graph from a phrase list and comment list 
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

def build_cooccurence_dict(comment_list, phrase_list, synonyms = [], graph = False, write = False, filename = None): 
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
        
        if write: 
            with open(filename ,'wb') as f: 
                pickle.dump(edge_dict, f)
                
        return sentence_list, edge_dict, G 

    else:
        
        if write: 
            with open(filename ,'wb') as f: 
                pickle.dump(edge_dict, f)
        
        return sentence_list, edge_dict     
    



##helpers#####
def color_nodes_by_type(graph, type_string):
    groups = set(nx.get_node_attributes(graph, type_string).values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = graph.nodes
    colors = [mapping[graph.nodes[n][type_string]] for n in nodes]

    return colors


def almost_disconnected(N, E, components = 2):
    '''create an almost disconnected graph where each component has N nodes and there are E edges between components'''

    G = nx.complete_graph(N)

    for i in range(components -1):
        G = nx.algorithms.operators.binary.disjoint_union(G, nx.complete_graph(N))
        edge_list = [(j, j+N) for j in range(i*N, i*N + E)]
        G.add_edges_from(edge_list)

    return G

