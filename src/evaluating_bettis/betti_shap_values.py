import random
import numpy as np
import pandas as pd
from igraph import *
import matplotlib.pyplot as plt
from ripser import ripser
import shap
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance

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
        train_diagrams = ripser(train_normalize, thresh=thresh, maxdim=0, distance_matrix=True)[
            'dgms']

        # splitting the dimension into 0 and 1
        train_dgm_0 = train_diagrams[0]

        # obtain betti numbers for the unique dimensions
        train_betti_0 = []

        for eps in np.linspace(0, 1, step_size):
            b_0 = 0
            for k in train_dgm_0:
                if k[0] <= eps and k[1] > eps:
                    b_0 = b_0 + 1
            train_betti_0.append(b_0)  # concatenate betti numbers

        train_betti.append(train_betti_0)
        file.write(dataset + "\t" + str(train_betti_0)[1:-1] + "\n")

        file.flush()

    train_data = pd.DataFrame(train_betti)
    #column_names = np.round(np.linspace(0, 1, step_size), 3)
    train_data.rename(columns={x: y for x, y in zip(train_data.columns, range(0,len(train_data.columns)))})

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

    plt.show()

    return x_train, x_test, y_test, clf, accuracy, cross_validation


# def vanilla_fi(train_data, x_test, y_test, clf):
# #    feature importance built-in the RandomForest algorithm (Mean Decrease in Impurity)
#     features = train_data.columns.values
#     feature_importance = clf.feature_importances_
#     std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
#     forest_importances = pd.Series(feature_importance, index=features)
#
#     fig, ax = plt.subplots()
#     forest_importances.plot.bar(yerr=std, ax=ax)
#     ax.set_title("Feature importances using MDI")
#     ax.set_ylabel("Mean decrease in impurity")
#     fig.tight_layout()
# #    plt.savefig('C:/XTDA-Paper/betti_shap_plots/FI_MDI2.png')
#     plt.clf()
#
# #    feature importance computed with permutation method
#     perm_importance = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
#     perm_importances = pd.Series(perm_importance.importances_mean, index=features)
#
#     fig, ax = plt.subplots()
#     perm_importances.plot.bar(yerr=perm_importance.importances_std, ax=ax)
#     ax.set_title("Feature importances using permutation on full model")
#     ax.set_ylabel("Mean accuracy decrease")
#     fig.tight_layout()
# #    plt.savefig('C:/XTDA-Paper/betti_shap_plots/FI_permutation2.png')
#
#     return


def shap_vals(clf, x_train):
#    feature importance computed with SHAP_values (Global Interpretability) (bar plot)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train)
#    print(explainer.expected_value) #the values obtained here serve as the base_values

    class_names = x_train.columns.values
    shap.summary_plot(shap_values, x_train, feature_names=class_names, show=False,
                      plot_size=(16, 10))
    # plt.savefig('C:/XTDA-Paper/betti_shap_plots/' + dataset + 'shap.png')
    # plt.clf()
#    shap.summary_plot(shap_values[i (for i= 0, 1 ..., n)], X_test, feature_names=features, show=False) #for dot plot. This can be generated for a single class(or observation) at a time

    return shap_values, class_names


def main():
    unique_graph_indicator, graph_indicators, df_edges, graph_labels = reading_csv()
    train_data = ripser_train(unique_graph_indicator, graph_indicators, df_edges)
    x_train, x_test, y_test, clf, accuracy, cross_validation = rf_model(train_data, graph_labels)
   # vanilla_fi(train_data, x_test, y_test, clf)
    shap_values, class_names = shap_vals(clf, x_train)


if __name__ == '__main__':
    data_path = sys.argv[1]  # dataset path on computer
    data_list = ('MUTAG', 'BZR') #, 'ENZYMES',  'PROTEINS', 'DHFR', 'NCI1', 'COX2')#, 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    outputFile = "C:/XTDA-Paper/results/" + 'A_.csv'  # "/home/taiwo/projects/def-cakcora/taiwo/result/" + 'Alpha_Pbetti.csv'
    file = open(outputFile, 'w')
    for dataset in data_list:
        for thresh in [1]:
            for step_size in [100]:  # we will consider stepsize 100 for epsilon
                main()

    file.close()
