import numpy as np
import pandas as pd
import random
from igraph import *
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import gudhi as gd
import gudhi.representations

random.seed(42)


def read_csv():

    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    print("Graph indicators are loaded")
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    y_train = (read_csv["ID"].values.astype(int))
    print("y_train is loaded")
    X_train = np.arange(min(graph_indicators), max(graph_indicators) + 1)  # list unique graph ids

    return X_train, graph_indicators, df_edges

def collect_graph(X_train, graph_indicators, df_edges):

    graph_id = X_train[3]
    id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                   element == graph_id]  # list the index of the graph_id locations
    graph_edges = df_edges[df_edges['from'].isin(id_location)]
    create_graph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
    plot(create_graph, "C:/Code/figures/mutag_graph.png")
    plt.clf()


    if not create_graph.is_connected():  # for the case of a disconnected graph
        graph_decompose = create_graph.decompose()
        mds_list = []
        for subg in graph_decompose:
            create_subg = np.asarray(Graph.shortest_paths_dijkstra(subg))
            norm_subg = create_subg / np.nanmax(create_subg)
            mds = MDS(n_components=2, dissimilarity='precomputed').fit_transform(
                norm_subg)  # reduce to two dimension
            mds_list.append(mds)
        matrix_mds = (np.vstack(mds_list))
    else:
        create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_graph))
        norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix)
        matrix_mds = MDS(n_components=2, dissimilarity='precomputed').fit_transform(norm_dmatrix)

    acX = gd.AlphaComplex(points=matrix_mds).create_simplex_tree()
    dgmX = acX.persistence()

    gd.plot_persistence_diagram(dgmX)  # plot persistence diagram
    plt.savefig("C:/Code/figures/" + "persistence_diagram.png")
    # plt.show()
    plt.clf()

    # plotting persistence landscape

    LS = gd.representations.Landscape(num_landscapes=1, resolution=1000)
    L = LS.fit_transform([acX.persistence_intervals_in_dimension(
        1)])  # returns persistence intervals of a simplicial complex in a specific dimension

    plt.plot(L[0][:1000])
    plt.box(False)
    plt.xlabel('t', fontdict={'weight':'bold'})
    plt.ylabel(r"$\lambda(k,t)$", fontdict={'weight':'bold'})
    # plt.plot(L[0][1000:2000])
    # plt.plot(L[0][2000:3000])
    plt.savefig("C:/Code/figures/" + "persistence_landscape.png")
    # plt.show()
    plt.clf()

    # plotting persistence silhouette

    SH = gd.representations.Silhouette(resolution=1000, weight=lambda x: 1)  # np.power(x[1] - x[0], 1))
    sh = SH.fit_transform([acX.persistence_intervals_in_dimension(1)])

    plt.plot(sh[0])
    plt.box(False)
    plt.xlabel('t', fontdict={'weight':'bold'})
    plt.ylabel(r"$\phi^{(i)}(t), i=1$", fontdict={'weight':'bold'})
    plt.savefig("C:/Code/figures/" + "persistence_silhouette.png")
    # plt.show()
    plt.clf()


def main():
    X_train, graph_indicators, df_edges = read_csv()
    collect_graph(X_train, graph_indicators, df_edges)

if __name__ == '__main__':
    data_path = sys.argv[1] #dataset path on computer
    data_list = ['MUTAG']
    for dataset in data_list:
        main()
