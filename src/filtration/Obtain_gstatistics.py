import random
import numpy as np
import pandas as pd
from igraph import *


def read_data():

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
                                       max(graph_indicators) + 1)  # list unique graph ids

    random.seed(42)

    graph_density = []
    graph_diameter = []
    clustering_coeff = []
    spectral_gap_ = []
    assortativity_ = []
    cliques = []
    motifs = []
    components = []
    graph_label_ = []

    for i in unique_graph_indicator:
        graph_id = i
        id_loc = [index + 1 for index, element in enumerate(graph_indicators) if
                  element == graph_id]  # list the index of the graphid locations
        graph_edges = df_edges[df_edges['from'].isin(id_loc)]  # obtain the edges with source node as train_graph_id
        graph_label = [v for u, v in enumerate(graph_labels, start=1) if u == i][0]
        set_graph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False,
                                    weights=True)  # obtain the graph

        Density = set_graph.density()  # obtain density
        Diameter = set_graph.diameter()  # obtain diameter
        cluster_coeff = set_graph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = set_graph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral_gap = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = set_graph.assortativity_degree()  # obtain assortativity
        clique_count = set_graph.clique_number()  # obtain clique count
        motifs_count = set_graph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(set_graph.clusters())  # obtain count components

        graph_label_.append(graph_label)
        graph_density.append(Density)
        graph_diameter.append(Diameter)
        clustering_coeff.append(cluster_coeff)
        spectral_gap_.append(spectral_gap)
        assortativity_.append(assortativity)
        cliques.append(clique_count)
        motifs.append(str(motifs_count)[1:-1])
        components.append(count_components)

    df = pd.DataFrame(
        data=zip(graph_density, graph_diameter, clustering_coeff, spectral_gap_, assortativity_, cliques, motifs,
                 components, graph_label_),
        columns=['graph_density', 'graph_diameter', 'clustering_coeff', 'spectral_gap', 'assortativity', 'cliques',
                 'motifs', 'components', 'graph_label'])

    df.insert(loc=0, column='dataset', value=dataset)
    #df.insert(loc=len(df.columns), column='graph_label', value=graph_label)

    return df


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    data_list = ('MUTAG', 'BZR', 'ENZYMES', 'BZR', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    df1 = []
    for dataset in data_list:
        df1.append(read_data())

    df2 = pd.concat(df1)

    df2.to_csv('C:/XTDA-Paper/results/A.csv')

   # df2.to_csv('/home/taiwo/projects/def-cakcora/taiwo/results3/Obtain_gstatistics.csv')
