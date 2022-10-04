import random
from datetime import datetime
from time import time
import numpy as np
from numpy import inf
import pandas as pd
from igraph import *
import gudhi as gd
from gudhi.representations import Silhouette
from sklearn.manifold import MDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

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

    X_train, X_test, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, test_size=0.2,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test, graph_indicators, df_edges, graph_labels


def landscape_train(X_train, graph_indicators, df_edges, step_size):  # this is for the train test
    start2 = time()
    train_silhouette = []
    for i in X_train:
        graph_id = i
        id_location = [index+1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)

        if not create_traingraph.is_connected():
            graph_decompose = create_traingraph.decompose()
            mds_list = []
            for subg in graph_decompose:
                create_subg = np.asarray(Graph.shortest_paths_dijkstra(subg))
                norm_subg = create_subg / np.nanmax(create_subg)
                mds = MDS(n_components=2, dissimilarity='precomputed').fit_transform(norm_subg)
                mds_list.append(mds)
            matrix_mds = (np.vstack(mds_list))
        else:
            create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
            norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix)
            matrix_mds = MDS(n_components=2, dissimilarity='precomputed').fit_transform(norm_dmatrix)

        train_alpha_complex = gd.AlphaComplex(points=matrix_mds)
        train_simplex_tree = train_alpha_complex.create_simplex_tree()
        train_diagrams = np.asarray(train_simplex_tree.persistence(), dtype='object')

       #select dimensions 0 and 1
        train_persist_0 = train_diagrams[:, 1][np.where(train_diagrams[:, 0] == 0)]
        train_persist_1 = train_diagrams[:, 1][np.where(train_diagrams[:, 0] == 1)]
        train_0 = np.array([list(x) for x in train_persist_0])
        train_1 = np.array([list(y) for y in train_persist_1])
        merge_array = [train_0, train_1]
        sample_range = np.linspace(0, 1, step_size)
        traingraph_silhouette = Silhouette(weight=lambda x: 1, resolution=len(sample_range),
                                          sample_range=[min(sample_range), max(sample_range)]).fit_transform(merge_array)
        train_silhouette.append(traingraph_silhouette[0] + traingraph_silhouette[1])

    t2 = time()
    train_time = t2 - start2

    return train_silhouette, train_time, sample_range


def landscape_test(X_test, graph_indicators, df_edges, sample_range, train_time, train_silhouette):
    start3 = time()
    test_silhouette = []
    for j in X_test:
        graph_id = j
        id_location = [index+1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_testgraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)

        if not create_testgraph.is_connected():
            graph_decompose = create_testgraph.decompose()
            mds_list = []
            for subg in graph_decompose:
                create_subg = np.asarray(Graph.shortest_paths_dijkstra(subg))
                norm_subg = create_subg / np.nanmax(create_subg)
                mds = MDS(n_components=2, dissimilarity='precomputed').fit_transform(norm_subg)
                mds_list.append(mds)
            matrix_mds = (np.vstack(mds_list))
        else:
            create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_testgraph))
            norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix)
            matrix_mds = MDS(n_components=2, dissimilarity='precomputed').fit_transform(norm_dmatrix)

        test_alpha_complex = gd.AlphaComplex(points=matrix_mds)  # initialize alpha complex
        test_simplex_tree = test_alpha_complex.create_simplex_tree()  # creating a simplex tree
        test_diagrams = np.asarray(test_simplex_tree.persistence(),
                                   dtype='object')
        test_persist_0 = test_diagrams[:, 1][np.where(test_diagrams[:, 0] == 0)]
        test_persist_1 = test_diagrams[:, 1][np.where(test_diagrams[:, 0] == 1)]
        test_0 = np.array([list(x) for x in test_persist_0])
        test_1 = np.array([list(y) for y in test_persist_1])
        merged_array = [test_0, test_1]
        testgraph_silhouette = Silhouette(weight=lambda x: 1, resolution=len(sample_range),
                                          sample_range=[min(sample_range), max(sample_range)]).fit_transform(merged_array)
        test_silhouette.append(testgraph_silhouette[0] + testgraph_silhouette[1])

    t3 = time()
    test_time = t3 - start3

    silhouette_time = train_time + test_time

    train_data = pd.DataFrame(train_silhouette)
    test_data = pd.DataFrame(test_silhouette)

    return train_data, test_data, silhouette_time


def tuning_hyperparameter():
    n_estimators = [int(a) for a in np.linspace(start=200, stop=500, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    Param_Grid = dict(n_estimators=n_estimators, max_depth=max_depth)

    return Param_Grid, num_cv


def random_forest(dataset, Param_Grid, train_data, test_data, y_train, y_test, silhouette_time, num_cv):
    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))
    start5 = time()
    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=Param_Grid, cv=num_cv, n_jobs=10)
    grid.fit(train_data, y_train)
    param_choose = grid.best_params_
    if len(set(y_test)) > 2:  # multiclass case
        print(dataset + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(train_data, y_train)
        y_pred = forest.predict(test_data)
        y_preda = forest.predict_proba(test_data)
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(train_data, y_train)
        test_pred = rfc_pred.predict(test_data)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(test_data)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
    print(dataset + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))

    t5 = time()
    training_time = t5 - start5

    print(f'Persistent took {silhouette_time} seconds, training took {training_time} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[
                    1:-1]  # flatten confusion matrix into a single row while removing the [ ]
    file.write(dataset + "\t" + str(silhouette_time) + "\t" + str(training_time) +
               "\t" + str(accuracy) + "\t" + str(auc)  + "\t" + str(flat_conf_mat) + "\n")

    file.flush()


def main():
    X_train, X_test, y_train, y_test, graph_indicators, df_edges, graph_labels = read_csv(dataset)
    train_silhouette, train_time, sample_range = landscape_train(X_train, graph_indicators, df_edges, step_size)
    train_data, test_data, silhouette_time = landscape_test(X_test, graph_indicators, df_edges, sample_range, train_time, train_silhouette)
    Param_Grid, num_cv = tuning_hyperparameter()
    random_forest(dataset, Param_Grid, train_data, test_data, y_train, y_test, silhouette_time, num_cv)


if __name__ == '__main__':
    data_path = "/home/taiwo/projects/def-cakcora/taiwo/data"	#dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K' )
    outputFile = "/home/taiwo/projects/def-cakcora/taiwo/results3/" + 'Alpha_MPS.csv'
    file = open(outputFile, 'w')
    for dataset in data_list:
        for step_size in [100]:  # we will consider step size 100 for epsilon
            for duplication in np.arange(5):
                main()
    file.close()

#
