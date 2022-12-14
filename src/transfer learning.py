import numpy as np
import pandas as pd
import random
import warnings
from igraph import *
from ripser import ripser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from time import time

warnings.filterwarnings("ignore", category=FutureWarning)

random.seed(42)


def read_csv1():

    df_edges = pd.read_csv(data_path + "/" + dataset1 + "/" + dataset1 + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset1 + "/" + dataset1 + "_graph_indicator.txt", header=None)
    print("Graph indicators are loaded")
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset1 + "/" + dataset1 + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    y_train = (read_csv["ID"].values.astype(int))
    print("y_train is loaded")
    X_train = np.arange(min(graph_indicators), max(graph_indicators) + 1)  # list unique graph ids

    return X_train, y_train, graph_indicators, df_edges


def read_csv2():

    df_edges2 = pd.read_csv(data_path + "/" + dataset2 + "/" + dataset2 + "_A.txt", header=None)  # import edge data
    df_edges2.columns = ['from', 'to']
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset2 + "/" + dataset2 + "_graph_indicator.txt", header=None)
    print("Graph indicators are loaded")
    csv.columns = ["ID"]
    graph_indicators2 = (csv["ID"].values.astype(int))
    read_csv = pd.read_csv(data_path + "/" + dataset2 + "/" + dataset2 + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    y_test = (read_csv["ID"].values.astype(int))
    print("y_test is loaded")
    X_test = np.arange(min(graph_indicators2), max(graph_indicators2) + 1)  # list unique graph ids

    return X_test, y_test, graph_indicators2, df_edges2


def ripser_train(X_train, thresh, graph_indicators, df_edges, step_size):  # this is for the train test
    start3 = time()
    train_betti = []
    for i in X_train:
        graph_id = i
        id_location = [index+1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        train_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        train_normalize = train_distance_matrix / np.nanmax(train_distance_matrix[train_distance_matrix != np.inf])
        train_diagrams = ripser(train_normalize, thresh=thresh, maxdim=1, distance_matrix=True)[
            'dgms']  # maximum homology dimension computed i.e H_0, H_1 for maxdim=1. thresh is maximum distance considered when constructing filtration
        #   plot_diagrams(train_diagrams, title= "train persistence diagrams showing H_0 and H_1",  show=True) #using thresh=1 because the largest number in the matrix is 1

        # splitting the dimension into 0 and 1
        train_persist_0 = train_diagrams[0]
        train_persist_1 = train_diagrams[1]

        # obtain the maximums
        try:
            max_1 = max(train_persist_0, key=lambda x: x[1] != np.inf)[1]
            max_2 = max(train_persist_1, key=lambda x: x[1] != np.inf)[1]
            common_max = max(max_1, max_2)
        except ValueError:
            common_max = max_1

        # obtain betti numbers for the unique dimensions
        train_betti_0 = []
        train_betti_1 = []

        for eps in np.linspace(0, common_max, step_size):
            b_0 = 0
            for k in train_persist_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            train_betti_0.append(b_0)

            b_1 = 0
            for l in train_persist_1:
                if l[0] <= eps and l[1] > eps:
                    b_1 = b_1 + 1
            train_betti_1.append(b_1)

        train_betti.append(train_betti_0 + train_betti_1)  # concatenate betti numbers

    t3 = time()
    ripser_time = t3 - start3

    return train_betti, ripser_time

def ripser_t(X_test, thresh, graph_indicators2, df_edges2, step_size, train_betti, ripser_time):  # this is for the test
    start4 = time()
    test_betti = []
    for j in X_test:
        graph_id = j
        id_location = [index+1 for index, element in enumerate(graph_indicators2) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges2[df_edges2['from'].isin(id_location)]
        create_testgraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        test_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_testgraph))
        test_normalize = test_distance_matrix / np.nanmax(test_distance_matrix[test_distance_matrix != np.inf])
        test_diagrams = ripser(test_normalize, thresh=thresh, maxdim=1, distance_matrix=True)[
            'dgms']  # using thresh=1 because the largest number in the matrix is 1
        # plt.clf()
        # plot_diagrams(test_diagrams, title= "test persistence diagrams showing H_0 and H_1",  show=True)

        # splitting the dimension into 0 and 1
        test_persist_0 = test_diagrams[0]
        test_persist_1 = test_diagrams[1]

        # obtain the maximums
        try:
            max_1 = max(test_persist_0, key=lambda x: x[1] != np.inf)[1]
            max_2 = max(test_persist_1, key=lambda x: x[1] != np.inf)[1]
            common_max = max(max_1, max_2)
        except ValueError:
            common_max = max_1

        # obtain betti numbers for the unique dimensions
        test_betti_0 = []
        test_betti_1 = []

        for eps in np.linspace(0, common_max, step_size):
            b_0 = 0
            for k in test_persist_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            test_betti_0.append(b_0)

            b_1 = 0
            for l in test_persist_1:
                if l[0] <= eps and l[1] > eps:
                    b_1 = b_1 + 1
            test_betti_1.append(b_1)

        test_betti.append(test_betti_0 + test_betti_1)  # concatenate betti numbers

    train_data = pd.DataFrame(train_betti)
    test_data = pd.DataFrame(test_betti)

    t4 = time()
    ripser_time_ = t4 - start4

    total_time = ripser_time + ripser_time_

    return train_data, test_data, total_time


def tuning_hyperparameter():
    n_estimators = [int(a) for a in np.linspace(start=200, stop=500, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    Param_Grid = dict(n_estimators=n_estimators, max_depth=max_depth)

    return Param_Grid, num_cv


def random_forest(Param_Grid, train_data, test_data, y_train, y_test, total_time, thresh, num_cv, step_size):
    start6 = time()

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=Param_Grid, cv=num_cv, n_jobs=10)
    grid.fit(train_data, y_train)
    param_choose = grid.best_params_
    rfc_pred = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(train_data, y_train)
    test_pred = rfc_pred.predict(test_data)
    auc = roc_auc_score(y_test, rfc_pred.predict_proba(test_data)[:, 1])
    accuracy = accuracy_score(y_test, test_pred)
    conf_mat = confusion_matrix(y_test, test_pred)

    t6 = time()
    training_time = t6 - start6

    print(f'Ripser took {total_time} seconds, training took {training_time} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[
                    1:-1]  # flatten confusion matrix into a single row while removing the [ ]

    file.write(str(total_time) + "\t" + str(training_time) + "\t" + str(accuracy) + "\t" + str(auc) + "\t" + str(flat_conf_mat) + "\n")

    file.flush()


def main():
    X_train, y_train, graph_indicators, df_edges = read_csv1()
    X_test, y_test, graph_indicators2, df_edges2 = read_csv2()
    train_betti, ripser_time = ripser_train(X_train, thresh, graph_indicators, df_edges, step_size)
    train_data, test_data, total_time = ripser_t(X_test, thresh, graph_indicators2, df_edges2, step_size, train_betti, ripser_time)
    Param_Grid, num_cv = tuning_hyperparameter()
    random_forest(Param_Grid, train_data, test_data, y_train, y_test, ripser_time, thresh, num_cv, step_size)


if __name__ == '__main__':
    data_path = sys.argv[1] #dataset path on computer
    data_1 = ['COX2']
    data_2 = ['BZR']
    outputFile = "C:/Code/results/" + 'transfer_learning1.csv'
    file = open(outputFile, 'w')
    for dataset1 in data_1:
        for dataset2 in data_2:
            for thresh in [1]:
                for step_size in [100]:  # we will consider step sizes [3,5,10,20,50] for epsilon
                    for duplication in np.arange(2):
                        main()
    file.close()

#
