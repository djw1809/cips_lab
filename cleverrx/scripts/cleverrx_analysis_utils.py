import numpy as np
import pandas as pd
import networkx as nx
import pickle
import heapq
from networkx.algorithms.operators.binary import disjoint_union, union
from sklearn.cluster import KMeans
from itertools import count
import matplotlib.pyplot as plt
import heapq
from mpl_toolkits.mplot3d import Axes3D



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

def recursive_automatic_ncut(G, steps, matrix_choice, num_v_start = 1, num_v_stop = 2, vlist = None, plot_graph = True, draw = False):

    total_ncut = 0
    clusterings = []
    normalized_ncut_values = []

    for i in range(len(G)):
        G.nodes[list(G.nodes)[i]]['label'] = 0

    clusterings.append(G)



    for i in range(steps):
        G_step = G.copy()
        n_cut_step = 0
        clusters = list(set(nx.get_node_attributes(clusterings[i], 'label').values()))
        for j in clusters:
            G_cluster = clusterings[i].subgraph([node for node in clusterings[i].nodes if clusterings[i].nodes[node]['label'] == j]).copy()
            #if len(G_cluster) == 1:
                #print('there is a 1 cluster likely failure')
            clustered_G = n_cut_classification(G_cluster, matrix_choice = 1)[0]
            label_0 = [node for node in clustered_G.nodes if clustered_G.nodes[node]['label'] == 0]
            label_1 = [node for node in clustered_G.nodes if clustered_G.nodes[node]['label'] == 1]
            n_cut_step += nx.algorithms.cuts.normalized_cut_size(clustered_G, label_0, label_1)

            for node in label_0:
                G_step.nodes[node]['label'] = max(list(set(nx.get_node_attributes(G_step, 'label').values())) + 1     #j*2^(i-1)

            for node in label_1:
                G_step.nodes[node]['label'] = max(list(set(nx.get_node_attributes(G_step, 'label').values())) + 1    #j*2^(i-1) + 1

        total_ncut += n_cut_step
        normalized_n_cut = np.divide(total_ncut, 2**(i+1))
        normalized_ncut_values.append(normalized_n_cut)
        clusterings.append(G_step)

        if draw:
            pos = nx.drawing.layout.spring_layout(G_step)
            edges = nx.draw_networkx_edges(G_step, pos, alpha = 0.2)
            colors = []
            node_labels = nx.get_node_attributes(G_step, 'label')
            for k in sorted(node_labels):
                colors.append(node_labels[k])
            colors = np.array(colors)
            nodes = nx.draw_networkx_nodes(G_step, pos, nodelist = G.nodes, node_color = colors, node_size = 50, cmap = plt.cm.jet, with_labels = False)
            plt.show()


    return clusterings, normalized_ncut_values







#
#
# def recursive_ncut(G, k, matrix_choice, num_v_start = 1, num_v_stop = 2, vlist = None, plot_graph = True):
#     '''recursive n-cut for classification tasks with k labels'''
#
#     G = nx.relabel.convert_node_labels_to_integers(G)
#     G_prune = G.copy()
#
#     labels = {}
#     for i in range(k-1):
#         lab = n_cut_classification(G_prune, matrix_choice, num_v_start, num_v_stop, vlist, plot_graph = plot_graph)[4]
#         keep = int(input('recluster red (1) or blue(0)'))
#         finished_labels = [m for m in range(len(lab)) if lab[m] != keep]
#         remove_nodes = []
#
#         for j in finished_labels:
#             labels[sorted(G.nodes)[j]] = i
#             remove_nodes.append(sorted(G.nodes)[j])
#
#         if i == k - 2:
#              other_finished_labels = [m for m in range(len(lab)) if lab[m] == keep]
#              for j in other_finished_labels:
#                  labels[sorted(G.nodes)[j]] = k
#
#         G_prune.remove_nodes_from(remove_nodes)
#
#
#
#     colors = [labels[i] for i in sorted(labels)]
#
#     if plot_graph:
#
#         pos = nx.drawing.layout.spring_layout(G)
#         nx.draw_networkx(G, pos = pos, node_color = colors, node_size = 200, cmap = plt.cm.jet)
#         #edges = nx.draw_networkx_edges(G, pos, alpha = 0.2)
#         #nodes = nx.draw_networkx_nodes(G, pos, nodelist = G.nodes, node_color = colors, node_size = 200, cmap = plt.cm.jet)
#
#         plt.show()
#
#     return G, colors, labels



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
