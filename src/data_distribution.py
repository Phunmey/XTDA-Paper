import random
import numpy as np
import pandas as pd
from igraph import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

random.seed(42)


def reading_csv():

    df_edges = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_A.txt", header=None)  # import edge data
    df_edges.columns = ['from', 'to']
    print("Graph edges are loaded")
    csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_indicator.txt", header=None)
    csv.columns = ["ID"]
    graph_indicators = (csv["ID"].values.astype(int))
    print("Graph indicators are loaded")
    read_csv = pd.read_csv(data_path + "/" + dataset + "/" + dataset + "_graph_labels.txt", header=None)
    read_csv.columns = ["ID"]
    graph_labels = (read_csv["ID"].values.astype(int))
    print("Graph labels are loaded")
    unique_graph_indicator = np.arange(min(graph_indicators),
                                       max(graph_indicators) + 1)  # list unique graph ids


    x_train, x_test, y_train, y_test = train_test_split(unique_graph_indicator, graph_labels, test_size=0.2,
                                                        random_state=42)


#   obtain the counts of classes and plot their distribution. Checking for class balance
    identify = ['y_train', 'y_test']
    unique, counts = np.unique(y_train, return_counts=True)
    plt.bar(unique, counts)
    unique_, counts_ = np.unique(y_test, return_counts=True)
    plt.bar(unique_, counts_)

    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.legend(identify)


    plt.savefig("C:/XTDA-Paper/class_distribution_plot/" + dataset + ".png")
    plt.clf()

    return




if __name__ == '__main__':
    data_path = sys.argv[1]
    data_list = ('ENZYMES', 'BZR', 'MUTAG', 'PROTEINS', 'DHFR', 'NCI1', 'COX2', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K')
    for dataset in data_list:
        reading_csv()
