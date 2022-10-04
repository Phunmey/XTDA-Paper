import pandas as pd
import numpy as np
from igraph import *
from collections import OrderedDict


def read_data(dataset, data_path):
    edges_asdf = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    edges_asdf.columns = ['from', 'to']
    print(dataset + " graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None) #returns a dataframe of graph indicators
    csv.columns = ["ID"]  # rename the column as ID
    graph_indicators = (csv["ID"].values.astype(int))
   # print(len(graph_indicators))

    unique_graph_indicator = np.arange(min(graph_indicators),max(graph_indicators) + 1)

    node_labels = []
    for graphid in unique_graph_indicator:
        graphid_loc1 = [index + 1 for index, element in enumerate(graph_indicators) if
                        element == graphid]  # list the index of the graphid locations
        edges_loc1 = edges_asdf[
            edges_asdf['from'].isin(graphid_loc1)]  # obtain edges that corresponds to these locations
        a_graph1 = Graph.TupleList(edges_loc1.itertuples(index=False), directed=False, weights=True)
        activation_values = list(a_graph1.degree())  # obtain node degrees
        collect_names = a_graph1.vs['name']
        create_dict = dict(zip(collect_names, activation_values))
        node_labels.append(create_dict)

    dict_nodelabels = [(k,v) for x in node_labels for k,v in x.items()]
    df_nodelabels = pd.DataFrame(dict_nodelabels, columns=['node_id', 'degree'])
    sort_nodelabels = df_nodelabels.sort_values(by=['node_id'])
    set_nodelabels = sort_nodelabels.set_index('node_id')
    nodelabels_list = (set_nodelabels['degree']).values.tolist()

    file.write('\n'.join(str(line) for line in nodelabels_list))

    file.flush()

if __name__ == '__main__':
    data_path = "/home/taiwo/projects/def-cakcora/taiwo/data"  # dataset path on computer
    data_list = ['REDDIT-MULTI-12K']
    outputFile = "/home/taiwo/projects/def-cakcora/taiwo/results3/" + 'REDDIT-MULTI-12K_node_labels.txt'
    file = open(outputFile, 'w')
    for dataset in data_list:
        read_data(dataset, data_path)
    file.close()