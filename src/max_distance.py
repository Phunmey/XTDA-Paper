import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

random.seed(42)

read_df = pd.read_csv("C:/XTDA-Paper/results/Obtain_gstatistics.csv")
read_df = read_df.fillna(0)

boxplot = read_df.boxplot(column='graph_diameter', by='dataset', rot=45, grid=False, figsize=(16, 12), fontsize=18)

#    remove all automated labels and titles
boxplot.get_figure().gca().set_xlabel("")
boxplot.get_figure().suptitle('')
boxplot.get_figure().gca().set_title("")

#    set labels
boxplot.set_xlabel('dataset', fontdict={'fontsize': 18})
boxplot.set_ylabel('graph diameter', fontdict={'fontsize': 18})
plt.tight_layout()
#plt.show()
plt.savefig("C:/XTDA-Paper/vectorization_plot/diameter_boxplot.png")


# sns.boxplot(x='dataset', y='graph_diameter', data=read_df, )
# plt.show()
