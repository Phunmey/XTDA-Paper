import sys
import matplotlib.pyplot as plt
import pandas as pd


def reading_csv():
    read_data = pd.read_csv(datapath + "/" + dataset + ".csv")
    print("data is loaded")

    for _, r in read_data.iterrows():
        plt.plot(r[['min', 'max']], [r['dataset'], r['dataset']])  # label=r['dataset'])
        plt.annotate(f"{r['min']}", (r['min'], r['dataset']), va='bottom')
        plt.annotate(f"{r['max']}", (r['max'], r['dataset']), va='bottom', ha='center')

    plt.tight_layout()
    #plt.show()
    plt.savefig("C:/XTDA-Paper/betti_shap_plots/" + dataset + "_minmax_.png")
    plt.clf()

    return

if __name__ == "__main__":
    datapath = sys.argv[1]
    datalist = ('betti0', 'betti1')
    for dataset in datalist:
        reading_csv()