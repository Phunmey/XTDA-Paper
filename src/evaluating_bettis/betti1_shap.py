import random
import numpy as np
import pandas as pd
from igraph import *
import matplotlib.pyplot as plt
from ripser import ripser
import shap
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

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
                                       max(graph_indicators) + 1)  # list unique graph ids

    return unique_graph_indicator, graph_indicators, df_edges, graph_labels


def ripser_train(unique_graph_indicator, graph_indicators, df_edges):  # this is for the train test

    train_betti = []
    for i in unique_graph_indicator:
        graph_id = i
        id_location = [index + 1 for index, element in enumerate(graph_indicators) if
                       element == graph_id]  # list the index of the graph_id locations
        graph_edges = df_edges[df_edges['from'].isin(id_location)]
        create_traingraph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False, weights=True)
        train_distance_matrix = np.asarray(Graph.shortest_paths_dijkstra(create_traingraph))
        train_normalize = train_distance_matrix / np.nanmax(train_distance_matrix[train_distance_matrix != np.inf])
        train_diagrams = ripser(train_normalize, thresh=thresh, maxdim=1, distance_matrix=True)[
            'dgms']

        # splitting the dimension into 0 and 1
        train_dgm_1 = train_diagrams[1]

        # obtain betti numbers for the unique dimensions
        train_betti_1 = []

        for eps in np.linspace(0, 1, step_size):
            b_1 = 0
            for k in train_dgm_1:
                if k[0] <= eps and k[1] > eps:
                    b_1 = b_1 + 1
            train_betti_1.append(b_1)  # concatenate betti numbers

        train_betti.append(train_betti_1)

        file.write(dataset + "\t" + str(train_betti_1)[1:-1] + "\n")

        file.flush()

    train_data = pd.DataFrame(train_betti)
    column_names = np.round(np.linspace(0, 1, step_size), 3)
    train_data = train_data.rename(columns={x: y for x, y in zip(train_data.columns, column_names)})

    return train_data


def rf_model(train_data, graph_labels):
    x_train, x_test, y_train, y_test = train_test_split(train_data, graph_labels, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=300).fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    #    model evaluation
    accuracy = metrics.accuracy_score(y_test, y_pred)  # obtain the accuracy score
    accuracies = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10)
    cross_validation = accuracies.mean()
    #    roc_score = roc_auc_score(y_test, clf.predict_proba(x_test))  # obtain roc_auc_score

    #    plt.show()

    return x_train, x_test, y_test, clf, accuracy, cross_validation


def shap_vals(clf, x_train):
    #    feature importance computed with SHAP_values (Global Interpretability) (bar plot)
    plt.clf()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train)
    #    print(explainer.expected_value) #the values obtained here serve as the base_values

    class_names = x_train.columns.values
    shap.summary_plot(shap_values, x_train, feature_names=class_names, show=False,
                      plot_size=(16, 10))
    plt.savefig('/project/def-cakcora/taiwo/betti_shap_plots/' + dataset + 'betti1shap.png')
    #    shap.summary_plot(shap_values[i (for i= 0, 1 ..., n)], X_test, feature_names=features, show=False) #for dot plot. This can be generated for a single class(or observation) at a time

    return


def main():
    unique_graph_indicator, graph_indicators, df_edges, graph_labels = reading_csv()
    train_data = ripser_train(unique_graph_indicator, graph_indicators, df_edges)
    x_train, x_test, y_test, clf, accuracy, cross_validation = rf_model(train_data, graph_labels)
    shap_vals(clf, x_train)


if __name__ == '__main__':
    data_path = "/home/taiwo/projects/def-cakcora/taiwo/data"  # dataset path on computer
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputFile = "/home/taiwo/projects/def-cakcora/taiwo/result/" + 'betti1_shap.csv'
    file = open(outputFile, 'w')
    for dataset in data_list:
        for thresh in [1]:
            for step_size in [100]:  # we will consider step size 100 for epsilon
                main()
    file.close()

