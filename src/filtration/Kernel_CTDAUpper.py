import random
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grakel.kernels import WeisfeilerLehman, VertexHistogram,CoreFramework
from igraph import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


def standardGraphFile(dataset, datapath):
    start1 = time()
    edges_asdf = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    edges_asdf.columns = ['from', 'to']
    unique_nodes = ((edges_asdf['from'].append(edges_asdf['to'])).unique()).tolist()
    print(dataset + " graph edges are loaded")
    csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    csv.columns = ["ID"]
    graphindicator = ((csv["ID"].values.astype(int)))
    print(dataset + " graph indicators are loaded")
    read_csv = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graphlabels_aslist = ((read_csv["ID"].values.astype(int)))
    print(dataset + " graph labels are loaded")
    read_nodelabel = pd.read_csv(datapath + "/" + dataset + "/" + dataset + "_node_labels.txt", header=None)
    read_nodelabel.columns = ["ID"]
    node_labels = (read_nodelabel["ID"].values.astype(int))
    print("Node labels are loaded")
    unique_graphindicator = np.arange(min(graphindicator),
                                      max(graphindicator) + 1)  # list unique graphids 100
    t1 = time()
    datatime = t1 - start1

    return unique_graphindicator, graphlabels_aslist, node_labels, graphindicator, edges_asdf, unique_nodes


def read_labels(node_labels, unique_nodes):
    nodes_dict = {}
    for index, ele in enumerate(node_labels):
        idx = index + 1
        if idx in unique_nodes:
            nodes_is = {idx: ele}
        else:
            unique_nodes.append(
                idx)  # if index is not found as node_id, append the index as a new node and give its corresponding node label
            nodes_is = {idx: ele}  # generated a random no

        nodes_dict.update(nodes_is)  # appending the dictionary to the outer dictionary

    return nodes_dict

def activation_discovery(dataset, edges_asdf, graphindicator, unique_graphindicator):
    # this function obtains the degrees(called activation_values) across all data and draws bar plots for them
    start2 = time()
    progress = len(unique_graphindicator)
    total_degree = {}
    node_degree_max = []  # list of the node degree maximums
    node_degree_min = []  # list of the node degree minimums
    print(dataset + " has " + str(progress) + " graphs.")
    for graphid in unique_graphindicator:
        if graphid % (progress / 100) == 0:
            print(str(graphid) + "/" + str(progress) + " completed")
        graphid_loc1 = [index + 1 for index, element in enumerate(graphindicator) if
                        element == graphid]  # list the index of the graphid locations
        edges_loc1 = edges_asdf[
            edges_asdf['from'].isin(graphid_loc1)]  # obtain edges that corresponds to these locations
        a_graph1 = Graph.TupleList(edges_loc1.itertuples(index=False), directed=False, weights=True)
        activation_values = np.asarray(a_graph1.degree())  # obtain node degrees
        # activation_values = [int(i) for i in np.asarray((a_graph1.betweenness()))] #obtain betweenness
        node_degree_max.append(max(activation_values))
        node_degree_min.append(min(activation_values))

        for i in activation_values:
            total_degree[i] = total_degree.get(i, 0) + 1
            # i is the key we want to return its value
            # dict.get(i,0) gives 0 if the key does not exist.
            # dict[i] is now assigned a value 1 (+1) which is an increment
    plt.bar(total_degree.keys(), total_degree.values(), color='b')  # 1 represents width
    plt.xticks(np.arange(min(node_degree_min), max(node_degree_max) + 1))
    plt.xlabel('Degrees')
    plt.ylabel('Fraction of nodes')  # obtained by dividing the node count of the filtration by the data node count
    plt.title(dataset)
    plt.savefig("/home/taiwo/projects/def-cakcora/taiwo/results3/degree_dist/" + dataset + "DegreeStats.png")
    print(dataset + " degree computations are completed.")

    max_activation = max(node_degree_max)  # obtain the maximum of the degree maximums
    min_activation = min(node_degree_min)  # obtain the minimum of the degree minimums
    print(dataset + " max activation was " + str(max(node_degree_max)) + ", we will use " + str(max_activation))
    print(dataset + " min activation was " + str(min(node_degree_min)))

    t2 = time()
    bartime = t2 - start2

    return max_activation, min_activation, progress


def filtration_discovery(dataset, filtration, h_filt, max_activation, max_allowed_filtration, min_activation):
    # this function computes the sublevel and superlevel filtrations
    start3 = time()
    if filtration == "sublevel":
        if h_filt:
            activation2 = int(max_activation / 2) + 1
            if (activation2 - min_activation) > max_allowed_filtration:
                filtr_range = np.unique(
                    np.linspace(start=min_activation, stop=activation2, dtype=int, num=max_allowed_filtration))
            else:
                filtr_range = np.arange(min_activation, activation2)
        else:
            if (max_activation - min_activation) > max_allowed_filtration:
                filtr_range = np.unique(
                    np.linspace(start=min_activation, stop=max_activation + 1, dtype=int,
                                num=max_allowed_filtration))
            else:
                filtr_range = np.arange(min_activation, max_activation + 1)
    else:
        if h_filt:
            activation3 = int(max_activation / 2) - 1
            if (max_activation - activation3) > max_allowed_filtration:
                filtr_range = np.flip(np.unique(
                    np.linspace(start=max_activation, stop=activation3, dtype=int, num=max_allowed_filtration)))
            else:
                filtr_range = np.arange(max_activation, activation3, -1)
        else:
            if (max_activation - min_activation) > max_allowed_filtration:
                filtr_range = np.flip(np.unique(
                    np.linspace(start=max_activation, stop=min_activation, dtype=int, num=max_allowed_filtration)))
            else:
                filtr_range = np.arange(max_activation, min_activation, -1)
    print(
        dataset + " filtration will run from " + str(filtr_range[0]) + " to " + str(
            filtr_range[len(filtr_range) - 1]))

    t3 = time()
    filttime = t3 - start3

    return filtr_range, filtration, h_filt


def kernelize_graph(unique_graphindicator, edges_asdf, filtr_range, filtration, dataset, graphindicator, progress,
                    nodes_dict, graphlabels_aslist):
    start4 = time()
    feature_matrix = []
    for graphid in unique_graphindicator:
        if graphid % (progress / 10) == 0:
            print(str(graphid) + "/" + str(progress) + " graphs completed")
        graphid_loc = [index + 1 for index, element in enumerate(graphindicator) if
                       element == graphid]  # list the index of the graphid locations
        edges_loc = edges_asdf[edges_asdf['from'].isin(graphid_loc)]  # obtain edges that corresponds to these locations
        a_graph = Graph.TupleList(edges_loc.itertuples(index=False), directed=False, weights=True)
        activation_values = np.asarray(a_graph.degree())
        # activation_values =[int(i) for i in np.asarray((a_graph.betweenness()))]

        wl_data = []
        for indx, deg in enumerate(filtr_range, start=0):
            if filtration == "sublevel":
                extract_vs = a_graph.vs.select([v for v, b in enumerate(activation_values) if
                                                b <= deg])  # returns vertexsequence that satisfies the "for" condition
                sub_graph = a_graph.subgraph(extract_vs)  # construct subgraphs from original graph using the indices
                subname = sub_graph.vs["name"]  # the subgraph vertex names
                subdict = {v: nodes_dict[v] for v in subname}
                subedges = list((edges_loc[edges_loc[['from', 'to']].isin(subname).all(1)]).to_records(index=False))
            else:
                extract_vs = a_graph.vs.select([v for v, b in enumerate(activation_values) if
                                                b >= deg])  # returns vertexsequence that satisfies the "for" condition
                sub_graph = a_graph.subgraph(extract_vs)  # construct subgraphs from original graph using the indices
                subname = sub_graph.vs["name"]  # the subgraph vertex names
                subdict = {v: nodes_dict[v] for v in subname}  # form dictionary with nodelabels
                subedges = list((edges_loc[edges_loc[['from', 'to']].isin(subname).all(1)]).to_records(
                    index=False))  # returns list of edges if both nodes are in subname

            if subedges == []:
                wl_data.append([{(0, 0)}, {0: 0}])  # file in this for all instances of []
            else:
                setedges = {tuple(item for item in pair) for pair in subedges}
                nodes_concat = [setedges, subdict]
                wl_data.append(nodes_concat)

        bk = (WeisfeilerLehman, dict(base_graph_kernel=VertexHistogram))
        ck = CoreFramework(base_graph_kernel=bk, min_core=-1, verbose=True)

        ck_transform = ck.fit_transform(wl_data)
        upper_diag = ck_transform[np.triu_indices(len(ck_transform), k=1)]
        feature_matrix.append(upper_diag)


    rfc_input = pd.DataFrame(feature_matrix)

    print(dataset + " has a feature matrix of " + str(rfc_input.shape))

    t4 = time()
    time_taken = t4 - start4


    random.seed(123)
    g_train, g_test, y_train, y_test = train_test_split(rfc_input, graphlabels_aslist, test_size=0.2,
                                                        random_state=random.randint(0, 100))

    return g_train, g_test, y_train, y_test, time_taken, t4


def rf_preprocess():
    max_features = ['auto', 'sqrt']
    n_estimators = [int(a) for a in np.linspace(start=300, stop=300, num=1)]
    max_depth = [int(b) for b in np.linspace(start=5, stop=6, num=2)]
    min_samples_split = [2, 5]
    min_samples_leaf = [2]
    bootstrap = [True, False]
    num_cv = 10
    gridlength = len(n_estimators) * len(max_depth) * len(min_samples_leaf) * len(min_samples_split) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    Param_Grid = dict(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, bootstrap=bootstrap)

    return Param_Grid, num_cv


def train_test_rf(Param_Grid, dataset, g_test, g_train, num_cv, y_test, y_train, time_taken, file, h_filt,
                  filtration):
    # start training
    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))
    start5 = time()
    rfc = RandomForestClassifier(n_jobs=10)
    grid = GridSearchCV(estimator=rfc, param_grid=Param_Grid, cv=num_cv, n_jobs=10)
    grid.fit(g_train, y_train)
    param_choose = grid.best_params_
    if len(set(y_test)) > 2:  # multiclass case
        print(dataset + " requires multi class RF")
        forest = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(g_train, y_train)
        y_pred = forest.predict(g_test)
        y_preda = forest.predict_proba(g_test)
        auc = roc_auc_score(y_test, y_preda, multi_class="ovr", average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        print(conf_mat)
    else:  # binary case
        rfc_pred = RandomForestClassifier(**param_choose, random_state=1, verbose=1).fit(g_train, y_train)
        test_pred = rfc_pred.predict(g_test)
        auc = roc_auc_score(y_test, rfc_pred.predict_proba(g_test)[:, 1])
        accuracy = accuracy_score(y_test, test_pred)
        conf_mat = confusion_matrix(y_test, test_pred)
        print(conf_mat)

    print(dataset + " accuracy is " + str(accuracy) + ", AUC is " + str(auc))

    t5 = time()
    rftime = t5 - start5

    print(f'Kernels took {time_taken} seconds, training took {rftime} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[1:-1]
    file.write(dataset + "\t" + str(time_taken) + "\t" + str(rftime) +
               "\t" + str(accuracy) + "\t" + str(auc)  + "\t" + str(filtration) + "\t" + str(
        h_filt) + "\t" +
               str(flat_conf_mat) + "\n")
    file.flush()

def main(h_filt, filtration, max_allowed_filtration):
    unique_graphindicator, graphlabels_aslist, node_labels, graphindicator, edges_asdf, unique_nodes = standardGraphFile(
        dataset, datapath)
    nodes_dict = read_labels(node_labels, unique_nodes)
    max_activation, min_activation, progress = activation_discovery(dataset, edges_asdf, graphindicator,
                                                                    unique_graphindicator)
    filtr_range, filtration, h_filt = filtration_discovery(dataset, filtration, h_filt, max_activation,
                                                           max_allowed_filtration, min_activation)
    g_train, g_test, y_train, y_test, time_taken, t4 = kernelize_graph(unique_graphindicator, edges_asdf,
                                                                       filtr_range,
                                                                       filtration, dataset, graphindicator,
                                                                       progress, nodes_dict, graphlabels_aslist)
    Param_Grid, num_cv = rf_preprocess()
    train_test_rf(Param_Grid, dataset, g_test, g_train, num_cv, y_test, y_train, time_taken, file, h_filt,
                  filtration)

if __name__ == '__main__':
    datapath = "/home/taiwo/projects/def-cakcora/taiwo/data"	#dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputFile = "/home/taiwo/projects/def-cakcora/taiwo/results3/" + 'Kernel_CTDAUpper.csv'
    file = open(outputFile, 'w')
    for dataset in data_list:
        for filtr_type in ('superlevel', 'sublevel'):
            for half in (True, False):
                for duplication in np.arange(10):
                    main(h_filt=half, filtration=filtr_type, max_allowed_filtration=100)
    file.close()


