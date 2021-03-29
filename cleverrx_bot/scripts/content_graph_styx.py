import networkx as nx
import json
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import graphviz
from graphviz import Graph
import itertools
from tqdm import tqdm
#%%

with open('../data/graph_new_combineddata/outputs/graph.pkl', 'rb') as file:
    data1 = pickle.load(file)

with open('../data/graph_final_final/outputs/fb_gaph_list_format.pkl', 'rb') as file:
    data2 = pickle.load(file)

with open('../data/topics_index_bots_new_042820.pkl', 'rb') as file:
    topic_links = pickle.load(file)

with open('../data/fb_graph.pkl', 'rb') as file:
    data4 = pickle.load(file)

#%%
topic_links
type = set()
bad_count = 0
for tweet in tqdm(topic_links.keys()):
    topic_link = topic_links[tweet]
    for link in topic_link['topic_links']:
        if len(link[4]) > 0:
            type.add(link[4])
        else:
            bad_count += 1
#%%
def draw_topic_link_graph(topic_links, filename):
    g = Graph('G', filename, engine = 'sfdp')
    g.attr(overlap = 'prism')
    g.attr(outputorder = 'edgesfirst')

    nodes_added = set()
    edgelist = []

    for tweet in tqdm(topic_links.keys()):
        topic_link = topic_links[tweet]['topic_links']
        nodes_in_tweet = set()
        for link in topic_link:
            if len(link[2]) == 0:
                node_name = link[1]+'####'+link[4].rstrip('+-')
                nodes_in_tweet.add(node_name.replace(':', ''))
            else:
                node_name = link[2]+'####'+link[4].rstrip('+-')
                nodes_in_tweet.add(node_name.replace(':', ''))

        edges = itertools.combinations(nodes_in_tweet, 2)

        for edge in edges:
            edgelist.append(edge)
            node1_name = edge[0]
            node2_name = edge[1]

            if node1_name not in nodes_added:
                g.node(node1_name, label = node1_name, style = 'filled', fontcolor = 'white', fontsize = str(40), fillcolor = 'black', overlap = "false", height = '5', width = '5')
                nodes_added.add(node1_name)

            if node2_name not in nodes_added:
                g.node(node2_name, label = node1_name, style = 'filled', fontcolor = 'white', fontsize = str(40), fillcolor = 'black', overlap = "false", height = '5', width = '5')
                nodes_added.add(node1_name)

            g.edge(node1_name, node2_name, style = 'solid', color = "#000000")
            edgelist.append(edge)


    return g, edgelist



color_config = {
"card":"#00308F",
"cost": "#9f0404",
"customers": "#e50606",
"health":"#F47A00",
"inhaler":"#055814",
"insurance":"#fcf928",
"medication":"#42f5b6",
"patients": "#F328B3",
"religion":"#8428F3",
"service": "#8AF328",
"None": "#C4CBBD"
}


def build_front_end_json(graph, edgelist, color_config):
    graphviz_json = graph.pipe('json').decode()
    graphviz_dict = json.loads(graphviz_json)




    selection_filters = [
        {
            "for": {},
            "filters": [
                {
                    "name": "1st Degree Neighboors",
                    "rule": "first_degree_nodes_selected",
                    "filter": []
                },
                {
                    "name": "2nd Degree Neighboors",
                    "rule": "second_degree_nodes_selected",
                    "filter": []
                },
                {
                    "name": "Graph",
                    "rule": "all-edges-fill-out",
                    "filter": []
                }
            ]
        }
    ]

    frontend_json = {"nodes": [], "links": [], "histograms": [], "config": {"legends": [], "selection_filters":selection_filters, "loading_settings": {}}}

    for key in color_config.keys():
        object = {"name":key,
                    "color":color_config[key],
                    "filter":[{"color":color_config[key]}]}

        frontend_json["config"]["legends"].append(object)

    nodes = graphviz_dict['objects']

    print("Adding nodes")
    for node in tqdm(nodes):
        node_object = {}

        name = node['name'].split('####')[0]
        type = node['name'].split('####')[1]
        if len(type) == 0:
            type = 'None'

        pos = node['pos'].split(',')
        node_object['x'] = pos[0]
        node_object['y'] = pos[1]
        node_object['id'] = name
        node_object["color"] = color_config[type]
        node_object['attributes'] = {"type":type}
        frontend_json['nodes'].append(node_object)

    print("adding edges")
    edge_count = 0

    for edge in tqdm(edgelist):
        edge_object = {}
        source = edge[0].split('####')[0]
        target = edge[1].split('####')[0]

        edge_object['source'] = source
        edge_object['target'] = target
        edge_object['id'] = edge_count
        edge_object['color'] = '#B9DF7F'
        edge_object['size'] = '1.0'

        frontend_json['links'].append(edge_object)
        edge_count +=1

    return frontend_json
#%%

g, edgelist = draw_topic_link_graph(topic_links, 'test')
edgelist = list(set(edgelist))
# graphviz_json = g.pipe('json').decode()
# graphviz_dict = json.loads(graphviz_json)
# nodes = graphviz_dict['objects']
# len(nodes)
final_json = build_front_end_json(g, edgelist, color_config)

with open('../data/styx_content_graph.json', 'w') as file:
    json.dump(final_json, file)
