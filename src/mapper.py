import webbrowser
import kmapper as km
import pandas as pd
import sklearn
from sklearn import manifold
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


# read data
statistics = "C:/XTDA-Paper/results/Obtain_gstatistics.csv"
merged_df = pd.read_csv(statistics, sep=',')

# split the motifs column into 4 different columns and drop columns
merged_df[["motif1", "motif2", "motif3", "motif4"]] = merged_df["motifs"].str.split(",", expand=True)
merged_df = merged_df.drop(["clustering_coeff", "graph_diameter", "motifs", "motif1", "motif2", "cliques", "components"], axis=1) #drop the index column from the dataframe

# group data based on the smallest data length which is mutag
df = merged_df.groupby(['dataset']).apply(lambda grp: grp.sample(n=187)) #mutag has 188 graphs

y = df[['dataset']]  # select dataset column as the label for mapper purpose
M = df[df.columns[1:]].apply(pd.to_numeric)  # convert object columns to numeric
M = M.drop(["graph_label"], axis=1)  # drop graph label column

# mapper process
Xfilt = M
cls = len(pd.unique(merged_df.iloc[:,0]))  # return length of unique elements in the first column
mapper = km.KeplerMapper()  # initialize mapper
scaler = MinMaxScaler(feature_range=(0, 1))  # scale fetaures
print(list(M.columns))
Xfilt = scaler.fit_transform(Xfilt)  # fit and transfrom features with scaler
lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE(verbose=1)) #dimensionality reduction of Xfilt
print("mapper started with "+str(len(pd.DataFrame(Xfilt).index))+" data points,"+str(cls)+" clusters")

graph = mapper.map(
    lens,
    Xfilt,
    clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
    cover=km.Cover(n_cubes=10, perc_overlap=0.6)
) # Create dictionary called 'graph' with nodes, edges and meta-information
print("mapper ended")
print(str(len(y))+" "+str(len(Xfilt)))

df['dataset_graphlabel'] = df['dataset'] + "-" + df['graph_label'].astype(str)
df = df.drop(["dataset","graph_label"], axis=1)
y_visual = df.dataset_graphlabel

html = mapper.visualize(
    graph,
    path_html="C:/XTDA-Paper/mapper_plot/mapper.html",
    title="mapper data",
    custom_tooltips=y_visual) # Visualize the graph

webbrowser.open('mapper.html')