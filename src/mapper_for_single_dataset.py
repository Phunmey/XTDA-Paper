import sys
import kmapper as km
import numpy as np
import pandas as pd
import sklearn
from igraph import *
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

# read data
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

    graph_density = []
    graph_diameter = []
    clustering_coeff = []
    spectral_gap_ = []
    assortativity_ = []
    cliques = []
    motifs = []
    components = []
    graph_label_ = []

    for i in unique_graph_indicator:
        graph_id = i
        id_loc = [index + 1 for index, element in enumerate(graph_indicators) if
                  element == graph_id]  # list the index of the graphid locations
        graph_edges = df_edges[df_edges['from'].isin(id_loc)]  # obtain the edges with source node as train_graph_id
        graph_label = [v for u, v in enumerate(graph_labels, start=1) if u == i][0]
        set_graph = Graph.TupleList(graph_edges.itertuples(index=False), directed=False,
                                    weights=True)  # obtain the graph

        Density = set_graph.density()  # obtain density
        Diameter = set_graph.diameter()  # obtain diameter
        cluster_coeff = set_graph.transitivity_avglocal_undirected()  # obtain transitivity
        laplacian = set_graph.laplacian()  # obtain laplacian matrix
        laplace_eigenvalue = np.linalg.eig(laplacian)
        sort_eigenvalue = sorted(np.real(laplace_eigenvalue[0]), reverse=True)
        spectral_gap = sort_eigenvalue[0] - sort_eigenvalue[1]  # obtain spectral gap
        assortativity = set_graph.assortativity_degree()  # obtain assortativity
        clique_count = set_graph.clique_number()  # obtain clique count
        motifs_count = set_graph.motifs_randesu(size=3)  # obtain motif count
        count_components = len(set_graph.clusters())  # obtain count components

        graph_label_.append(graph_label)
        graph_density.append(Density)
        graph_diameter.append(Diameter)
        clustering_coeff.append(cluster_coeff)
        spectral_gap_.append(spectral_gap)
        assortativity_.append(assortativity)
        cliques.append(clique_count)
        motifs.append(str(motifs_count)[1:-1])
        components.append(count_components)

    df = pd.DataFrame(
        data=zip(graph_density, graph_diameter, clustering_coeff, spectral_gap_, assortativity_, cliques, motifs,
                 components, graph_label_),
        columns=['graph_density', 'graph_diameter', 'clustering_coeff', 'spectral_gap', 'assortativity', 'cliques',
                 'motifs', 'components', 'graph_label'])

    df.insert(loc=0, column='dataset', value=dataset)

    return df


def read_statistics(df):

    #    split the motifs column into 4 different columns and drop columns
    df[["motif1", "motif2", "motif3", "motif4"]] = df["motifs"].str.split(",", expand=True)
    df = df.drop(
        ["clustering_coeff", "graph_diameter", "motifs", "motif1", "motif2", "cliques", "components"],
        axis=1)  # drop the index column from the dataframe
    df = df.fillna(0)

    # group data based on the smallest data length which is mutag
    # df = merged_df.groupby(['dataset']).apply(lambda grp: grp.sample(n=188))  # mutag has 188 graphs

    y = df[['dataset']]  # select dataset column as the label for mapper purpose
    M = df[df.columns[1:]].apply(pd.to_numeric)  # convert object columns to numeric
    M = M.drop(["graph_label"], axis=1)  # drop graph label column

    # mapper process
    Xfilt = M  # input data
    cls = len(pd.unique(df.iloc[:, 0]))  # return length of unique elements in the first column of merged_df
    mapper = km.KeplerMapper()  # initialize mapper
    scaler = MinMaxScaler(feature_range=(0, 1))  # initialize scaler to scale features
    print(list(M.columns))
    Xfilt = scaler.fit_transform(Xfilt)  # fit and transfrom features with scaler
    lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE(verbose=1))  # dimensionality reduction of Xfilt
    print("mapper started with " + str(len(pd.DataFrame(Xfilt).index)) + " data points," + str(cls) + " clusters")

    graph = mapper.map(
        lens,
        Xfilt,
        #clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
        clusterer=sklearn.cluster.DBSCAN(),
        cover=km.Cover(n_cubes=10, perc_overlap=0.3)
    )  # Create dictionary called 'graph' with nodes, edges and meta-information
    print("mapper ended")
    print(str(len(y)) + " " + str(len(Xfilt)))

    df['dataset_graphlabel'] = df['dataset'] + "-" + df['graph_label'].astype(str)
    df = df.drop(["dataset", "graph_label"], axis=1)
    y_visual = df.dataset_graphlabel

    html = mapper.visualize(
        graph,
        path_html="C:/XTDA-Paper/mapper_plot/single_mapper.html",
        title="mapper data",
        custom_tooltips=y_visual)  # Visualize the graph

def main():
    df = reading_csv()
    read_statistics(df)


if __name__ == '__main__':
    data_path = sys.argv[1] #dataset path on computer
    data_list = ['PROTEINS']
    for dataset in data_list:
        main()