import random
import numpy as np
import pandas as pd
from igraph import *
import gudhi as gd
import matplotlib.pyplot as plt
from ripser import ripser
import gudhi.representations
from persim import plot_diagrams
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids

    return unique_graph_indicator, graph_indicators, df_edges


def dgms_calc(unique_graph_indicator, graph_indicators, df_edges):  # this is for the train test
    dgms_0 = []
    dgms_1 = []
    for i in unique_graph_indicator:
        graph_id = i
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_graph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        dist_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_graph))
        mat_normalize = dist_matrix / np.nanmax(dist_matrix[dist_matrix != np.inf])
        pers_diagrams = ripser(mat_normalize, thresh=1, maxdim=1, distance_matrix=True)[
            'dgms']  # maximum homology dimension computed i.e H_0, H_1 for maxdim=1. thresh is maximum distance considered when constructing filtration
        # plot_diagrams(pers_diagrams, show=True)
        dgms_0.append(pers_diagrams[0])
        dgms_1.append(pers_diagrams[1])

    #    remove infinity values in the list of persistence diagrams[0]
    remove_infinity = lambda pdgm: np.array([dots for dots in pdgm if dots[1] != np.inf])
    dgms_0 = list(map(remove_infinity, dgms_0))
    dgms_1 = list(map(remove_infinity, dgms_1))

    return dgms_0, dgms_1


def entropy_calc(dgms_0, dgms_1):
    for j in np.arange(len(dgms_1)):
        if (dgms_1[j]).size == 0:
            dgms_1[j] = np.array([[0, 0], [0, 0]])

    pe_init = gd.representations.Entropy()
    pe_0 = pe_init.fit_transform(dgms_0)  # persistent entropy for H_0
    pe_1 = pe_init.fit_transform(dgms_1)  # persistent entropy for H_1

    #    concatenate the arrays obtained for pe_0 and pe_1
    array_conc = np.concatenate((pe_0, pe_1), axis=1)
    df = pd.DataFrame(array_conc, columns=['entropy0', 'entropy1'])  # convert to df

    df.insert(loc=0, column='dataset', value=dataset)  # insert a column of data names

    df.to_csv(outputFile, mode='a', header=not os.path.isfile(outputFile), index=False)

    # if not os.path.isfile(outputFile):
    #     df.to_csv(outputFile, header=['dataset', 'entropy0', 'entropy1'], index=False)
    # else:
    #     df.to_csv(outputFile, mode='a', header=False, index=False)

   # file.flush()

    return


def main():
    unique_graph_indicator, graph_indicators, df_edges = reading_csv()
    dgms_0, dgms_1 = dgms_calc(unique_graph_indicator, graph_indicators, df_edges)
    entropy_calc(dgms_0, dgms_1)


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    data_list = ('ENZYMES', 'MUTAG', 'BZR', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputFile = "C:/XTDA-Paper/src/entropy/" + 'entropy_value.csv'
    for dataset in data_list:
        main()

