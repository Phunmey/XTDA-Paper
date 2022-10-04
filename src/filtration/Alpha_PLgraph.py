import random
from datetime import datetime
from time import time
import numpy as np
import pandas as pd
from igraph import *
import gudhi as gd
import gudhi.representations
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


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

    x_train, x_test, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, test_size=0.2,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test, graph_indicators, df_edges, graph_labels


def landscape_train(x_train, graph_indicators, df_edges):  # this is for the train data
    start2 = time()

    train_landscape = []
    graph_density = []
    graph_diameter = []
    clustering_coeff = []
    spectral_gap = []
    assortativity_ = []
    cliques = []
    motifs = []
    components = []

    for i in x_train:
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
                mds = PCA(n_components=2).fit_transform(norm_subg)
                mds_list.append(mds)
            matrix_mds = (np.vstack(mds_list))
        else:
            create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
            norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix)
            matrix_mds = PCA(n_components=2).fit_transform(norm_dmatrix)

        train_ac = gd.AlphaComplex(points=matrix_mds).create_simplex_tree()
        train_dgm = train_ac.persistence()  # obtain persistence values
    #    gd.plot_persistence_diagram(train_dgm)
    #    plt.show()

    #    select dimensions 0 and 1
    #    train_dgm_0 = train_ac.persistence_intervals_in_dimension(0)
        train_dgm_1 = train_ac.persistence_intervals_in_dimension(1)

    #    obtain persistence landscape values
        landscape_init = gd.representations.Landscape(num_landscapes=1, resolution=1000)
        land_scape = landscape_init.fit_transform([train_dgm_1])
    #    plt.plot(land_scape[0][:1000])
    #    plt.show()

        train_landscape.append(land_scape)


        Density = create_traingraph.density()  # obtain density
        Diameter = create_traingraph.diameter()  # obtain diameter
        cluster_coeff = create_traingraph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = create_traingraph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = create_traingraph.assortativity_degree()  # obtain assortativity
        clique_count = create_traingraph.clique_number()  # obtain clique count
        motifs_count = create_traingraph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(create_traingraph.clusters())  # obtain count components

        graph_density.append(Density)
        graph_diameter.append(Diameter)
        clustering_coeff.append(cluster_coeff)
        spectral_gap.append(spectral)
        assortativity_.append(assortativity)
        cliques.append(clique_count)
        motifs.append(motifs_count)
        components.append(count_components)

    df1 = pd.DataFrame(np.concatenate(train_landscape))
    df2 = pd.DataFrame(motifs)
    df3 = pd.DataFrame(list(zip(graph_density, graph_diameter, clustering_coeff, spectral_gap, assortativity_, cliques, components)))
    train_data = pd.concat([df1, df2, df3], axis=1, ignore_index=True)
    train_data = train_data.fillna(0)

    t2 = time()
    train_time = t2 - start2

    return train_data, train_time


def landscape_test(x_test, graph_indicators, df_edges, train_time):  # this is for the train test
    start3 = time()

    test_landscape = []
    test_graph_density = []
    test_graph_diameter = []
    test_clustering_coeff = []
    test_spectral_gap = []
    test_assortativity_ = []
    test_cliques = []
    test_motifs = []
    test_components = []

    for j in x_test:
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
                mds = PCA(n_components=2).fit_transform(norm_subg)
                mds_list.append(mds)
            matrix_mds = (np.vstack(mds_list))
        else:
            create_dmatrix = np.asarray(Graph.shortest_paths_dijkstra(create_testgraph))
            norm_dmatrix = create_dmatrix / np.nanmax(create_dmatrix)
            matrix_mds = PCA(n_components=2).fit_transform(norm_dmatrix)

        test_ac = gd.AlphaComplex(points=matrix_mds).create_simplex_tree()
        test_dgm = test_ac.persistence()  # obtain persistence values
    #    gd.plot_persistence_diagram(train_dgm)
    #    plt.show()

    #    select dimensions 0 and 1
    #    test_dgm_0 = test_ac.persistence_intervals_in_dimension(0)
        test_dgm_1 = test_ac.persistence_intervals_in_dimension(1)

    #    obtain persistence landscape values
        landscape_init = gd.representations.Landscape(num_landscapes=1, resolution=1000)
        land_scape = landscape_init.fit_transform([test_dgm_1])
    #    plt.plot(land_scape[0][:1000])
    #    plt.show()

        test_landscape.append(land_scape)

        Density = create_testgraph.density()  # obtain density
        Diameter = create_testgraph.diameter()  # obtain diameter
        cluster_coeff = create_testgraph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = create_testgraph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = create_testgraph.assortativity_degree()  # obtain assortativity
        clique_count = create_testgraph.clique_number()  # obtain clique count
        motifs_count = create_testgraph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(create_testgraph.clusters())  # obtain count components

        test_graph_density.append(Density)
        test_graph_diameter.append(Diameter)
        test_clustering_coeff.append(cluster_coeff)
        test_spectral_gap.append(spectral)
        test_assortativity_.append(assortativity)
        test_cliques.append(clique_count)
        test_motifs.append(motifs_count)
        test_components.append(count_components)

    df1_ = pd.DataFrame(np.concatenate(test_landscape))
    df2_ = pd.DataFrame(test_motifs)
    df3_ = pd.DataFrame(
        list(zip(test_graph_density, test_graph_diameter, test_clustering_coeff, test_spectral_gap, test_assortativity_, test_cliques, test_components)))
    test_data = pd.concat([df1_, df2_, df3_], axis=1, ignore_index=True)
    test_data = test_data.fillna(0)

    t3 = time()
    test_time = t3 - start3

    landscape_time = train_time + test_time

    return test_data, landscape_time


def tuning_hyperparameter():
    n_estimators = [int(a) for a in np.linspace(start=200, stop=500, num=5)]
    max_depth = [int(b) for b in np.linspace(start=2, stop=10, num=6)]
    num_cv = 10
    bootstrap = [True, False]
    gridlength = len(n_estimators) * len(max_depth) * num_cv
    print(str(gridlength) + " RFs will be created in the grid search.")
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)

    return param_grid, num_cv


def random_forest(param_grid, train_data, test_data, y_train, y_test, landscape_time, num_cv):
    print(dataset + " training started at", datetime.now().strftime("%H:%M:%S"))
    start5 = time()

    rfc = RandomForestClassifier()
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=num_cv, n_jobs=10)
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

    print(f'Alpha took {landscape_time} seconds, training took {training_time} seconds')
    flat_conf_mat = (str(conf_mat.flatten(order='C')))[
                    1:-1]  # flatten confusion matrix into a single row while removing the [ ]
    file.write(dataset + "\t" + str(landscape_time) + "\t" + str(training_time) +
               "\t" + str(accuracy) + "\t" + str(auc) + "\t" + str(flat_conf_mat) + "\n")

    file.flush()


def main():
    x_train, x_test, y_train, y_test, graph_indicators, df_edges, graph_labels = reading_csv()
    train_data, train_time = landscape_train(x_train, graph_indicators, df_edges)
    test_data, landscape_time = landscape_test(x_test, graph_indicators, df_edges, train_time)
    param_grid, num_cv = tuning_hyperparameter()
    random_forest(param_grid, train_data, test_data, y_train, y_test, landscape_time, num_cv)


if __name__ == '__main__':
    data_path = "/home/taiwo/projects/def-cakcora/taiwo/data"	#dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputFile = "/home/taiwo/projects/def-cakcora/taiwo/result/" + 'Alpha_PLgraph.csv'
    file = open(outputFile, 'w')
    for dataset in data_list:
        for duplication in np.arange(5):
            main()
    file.close()

#
