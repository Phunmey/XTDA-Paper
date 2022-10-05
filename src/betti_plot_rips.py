import random
from time import time
import numpy as np
import pandas as pd
from igraph import *
import matplotlib.pyplot as plt
from ripser import ripser
from sklearn.preprocessing import normalize
import seaborn as sns; sns.set(style='white')

random.seed(42)


def read_csv(dataset):
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

    return unique_graph_indicator, graph_indicators, df_edges, graph_labels


def ripser_train(unique_graph_indicator, thresh, graph_indicators, df_edges, step_size, dataset):  # this is for the train test
    start2 = time()
    train_betti = []
    for i in unique_graph_indicator:
        graph_id = i
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        train_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        train_normalize = train_distance_matrix / np.nanmax(train_distance_matrix[train_distance_matrix != np.inf])
        train_diagrams = ripser(train_normalize, thresh=thresh, maxdim=2, distance_matrix=True)[
            'dgms']

        # splitting the dimension into 0 and 1
        train_dgm_0 = train_diagrams[0]
        train_dgm_1 = train_diagrams[1]
        train_dgm_2 = train_diagrams[2]

        # obtain betti numbers for the unique dimensions
        train_betti_0 = []
        train_betti_1 = []
        train_betti_2 = []

        for eps in np.linspace(0, 1, step_size):
            b_0 = 0
            for k in train_dgm_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            train_betti_0.append(b_0)  # concatenate betti numbers

            b_1 = 0
            for l in train_dgm_1:
                if l[0] <= eps and l[1] > eps:
                    b_1 = b_1 + 1
            train_betti_1.append(b_1)

            b_2 = 0
            for m in train_dgm_2:
                if m[0] <= eps and m[1] > eps:
                    b_2 = b_2 + 1
            train_betti_2.append(b_2)

        norm_0 = normalize([train_betti_0], norm="max")
        norm_1 = normalize([train_betti_1], norm="max")
        norm_2 = normalize([train_betti_2], norm="max")

        conc = np.concatenate((norm_0, norm_1, norm_2), axis=1)

        train_betti.append(conc)

       # train_betti.append(train_betti_0 + train_betti_1 + train_betti_2)

    train_data = pd.DataFrame(np.concatenate(train_betti))
#    summarized_df = train_data.describe()

    obtain_mean = train_data.mean(axis=0)
#    obtain_mean = summarized_df.loc['mean']
#    quartiles1 = summarized_df.loc['25%']
#    quartiles3 = summarized_df.loc['75%']

    x = np.arange(0, 100)

    betti0 = obtain_mean.iloc[:100]
    betti1 = obtain_mean.iloc[100:200]
    betti2 = obtain_mean.iloc[200:]

    sns.lineplot(x=x, y=betti0)
    sns.lineplot(x=x, y=betti1)
    sns.lineplot(x=x, y=betti2)

    plt.box(False)
    plt.xlabel(r"$\epsilon$")
    plt.ylabel('mean frequency')
    plt.legend(labels=['B0', 'B1', 'B2'])



    # ax0.fill_between(x, quartiles1[:100], quartiles3[:100], alpha=0.3)
    # ax1.fill_between(x, quartiles1[100:200], quartiles3[100:200], alpha=0.3)
    # ax2.fill_between(x, quartiles1[200:], quartiles3[200:], alpha=0.3)
    plt.show()
    plt.clf()


    plt.savefig("C:/XTDA-Paper/betti_plots_rips/" + dataset + ".png")
    plt.clf()

    return


def main():
    unique_graph_indicator, graph_indicators, df_edges, graph_labels = read_csv(dataset)
    ripser_train(unique_graph_indicator, thresh, graph_indicators, df_edges, step_size, dataset)


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    data_list = ('MUTAG', 'ENZYMES', 'BZR', 'PROTEINS', 'DHFR', 'NCI1', 'COX2')#, 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        for thresh in [1]:
            for step_size in [100]:  # we will consider stepsize 100 for epsilon
                main()

#
