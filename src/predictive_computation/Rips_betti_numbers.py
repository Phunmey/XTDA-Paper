import random
import sys
import numpy as np
import pandas as pd
from igraph import *
from ripser import ripser
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

random.seed(42)


def reading_csv():
    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    print("Graph indicators are loaded")
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    print("Graph labels are loaded")
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids in a dataset

    return unique_graph_indicator, graph_indicators, df_edges, graph_labels


def betti_stats(unique_graph_indicator, graph_indicators, df_edges):  # this is for the train test

    bettis = {}
    for i in unique_graph_indicator:
        graph_id = i
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_graph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        train_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_graph))
        train_normalize = train_distance_matrix / np.nanmax(train_distance_matrix[train_distance_matrix != np.inf])
        train_rips = ripser(train_normalize, thresh=1, maxdim=2, distance_matrix=True)[
            'dgms']

        B0 = 0 if len(train_rips[0]) == 0 else 1
        B1 = 0 if len(train_rips[1]) == 0 else 1
        B2 = 0 if len(train_rips[2]) == 0 else 1

        density = create_graph.density()  # obtain density
        diameter = create_graph.diameter()  # obtain diameter
        cluster_coeff = create_graph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = create_graph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral_gap = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = create_graph.assortativity_degree()  # obtain assortativity
        clique_count = create_graph.clique_number()  # obtain clique count
        motifs_count = create_graph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(create_graph.clusters())  # obtain count components

        bettis.update({str(graph_id): [density, diameter, cluster_coeff, spectral_gap, assortativity, clique_count,
                                       motifs_count, count_components, B0, B1, B2]})  # append to external dictionary

    df = (pd.DataFrame.from_dict(bettis, orient='index',
                                 columns=['graph_density', 'graph_diameter', 'clustering_coeff', 'spectral_gap',
                                          'assortativity', 'cliques', 'motifs', 'components', 'BO', 'B1',
                                          'B2'])).rename_axis('id').reset_index()

    df.to_csv("C:/XTDA-Paper/src/predictive_computation/" + dataset + "_betti.csv")


def main():
    unique_graph_indicator, graph_indicators, df_edges, graph_labels = reading_csv()
    betti_stats(unique_graph_indicator, graph_indicators, df_edges)


if __name__ == '__main__':
    data_path = sys.argv[1]
    data_list = ('MUTAG', 'BZR', 'ENZYMES', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        main()
