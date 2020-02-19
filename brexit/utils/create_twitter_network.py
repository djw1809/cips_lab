import networkx as nx
import database_connect_query as d


def create_retweet_edgelist(number_of_edges, filename, write = True):
    G = nx.DiGraph()

    conn = d.get_db_connection()
    cursor = conn.cursor()

    results = d.get_retweet_edge_list(conn, number_of_edges)

    for row in results:
        user_from = int(row[2])
        user_to = int(row[3])

        if G.has_edge(user_from, user_to):
            G[user_from][user_to]['weight'] += 1

        else:
            G.add_edge(user_from, user_to, weight = 1)

    if write:
        nx.drawing.nx_pydot.write_dot(G, filename)

    return results, G
