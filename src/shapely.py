import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import shap
import sys
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

random.seed(42)

def data_csv(datapath):
    merged_df = pd.read_csv(datapath, sep=",")
    read_data = merged_df.fillna(0)

#    split the motifs column into 4 different columns and drop columns
    read_data[["motif1", "motif2", "motif3", "motif4"]] = read_data["motifs"].str.split(",", expand=True)
    read_data = read_data.drop(["graph_label", "motifs", "motif1", "motif2"], axis=1)  # drop the index column from the dataframe

    x = read_data[read_data.columns[1:]]
#    X = X.fillna(0)  # Features
    y = read_data['dataset']  # Labels

    return x, y

def rf_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=300).fit(x_train, y_train)
    y_pred = clf.predict(x_test)

#    model evaluation
    accuracy = metrics.accuracy_score(y_test, y_pred)  # obtain the accuracy score
    accuracies = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10)
    cross_validation = accuracies.mean()
    roc_score = roc_auc_score(y_test, clf.predict_proba(x_test), multi_class='ovr')  # obtain roc_auc_score
#   plot_confusion_matrix(clf, X_test, y_test, display_labels=features, cmap=plt.cm.Blues, xticks_rotation='vertical' )
#   plt.show()
    return x_train, x_test, y_test, clf, accuracy, cross_validation, roc_score

def vanilla_fi(X, X_test, y_test, clf):
#    feature importance built-in the RandomForest algorithm (Mean Decrease in Impurity)
    features = X.columns.values
    feature_importance = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(feature_importance, index=features)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("feature importances using mdi")
    ax.set_ylabel("mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('C:Code/other_figures/fi_mdi2.png')
    #
    # #feature importance computed with permutation method
    perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    perm_importances = pd.Series(perm_importance.importances_mean, index=features)

    fig, ax = plt.subplots()
    perm_importances.plot.bar(yerr=perm_importance.importances_std, ax=ax)
    ax.set_title("feature importances using permutation on full model")
    ax.set_ylabel("mean accuracy decrease")
    fig.tight_layout()
    plt.savefig('C:Code/other_figures/fi_permutation2.png')

    return features

def shap_vals(clf, x_train, features):
    # feature importance computed with SHAP_values (Global Interpretability) (bar plot)
    plt.clf()
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(x_train[:5])
    # print(explainer.expected_value) #the values obtained here serve as the base_values
    class_names = ["ENZYMES", "BZR", "COX2", "DHFR", "MUTAG", "NCI1", "PROTEINS", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K"]
    shap.summary_plot(shap_values, x_train, feature_names=features, class_names=class_names, show=False,
                      plot_size=(16, 10))
    plt.savefig('C:Code/other_figures/shap_dataset.png')
# shap.summary_plot(shap_values[i (for i= 0, 1 ..., n)], X_test, feature_names=features, show=False) #for dot plot. This can be generated for a single class(or observation) at a time

    return shap_values, class_names

def shap_feature_ranking(x_train, shap_values, features, class_names):
    c_idxs = []
    idx = [c_idxs.append(x_train.columns.get_loc(column)) for column in features] # Get column locations for desired columns in given dataframe
    if isinstance(shap_values, list):  # If shap values is a list of arrays (i.e., several classes)
        means = [np.abs(shap_values[class_][:, c_idxs]).mean(axis=0) for class_ in
                 range(len(shap_values))]  # Compute mean shap values per class
        stds = [np.abs(shap_values[class_][:, c_idxs]).std(axis=0) for class_ in
                 range(len(shap_values))]  # Compute standard deviation of shap values per class
        shap_means = np.sum(np.column_stack(means), 1)  # Sum of shap values over all classes
        shap_stds = np.sum(np.column_stack(stds), 1)  # Sum of shap values over all classes
    else:  # Else there is only one 2D array of shap values
        assert len(shap_values.shape) == 2, 'Expected two-dimensional shap values array.'
        shap_means = np.abs(shap_values).mean(axis=0)

    collect_mean = (pd.DataFrame(means, columns=features)).round(decimals=3)
    collect_std = (pd.DataFrame(stds, columns=features)).round(decimals=3)

    shap_importance = (pd.DataFrame(list(zip(class_names, features, means, stds)),columns=['class_name','feature_name', 'mean_of_feature', 'std_of_feature'])).round(decimals=3)

    # Put into dataframe along with columns and sort by shap_means, reset index to get ranking
    df_ranking = shap_importance.sort_values(by=['mean_of_feature'],ascending=False)
   # df_ranking.index += 1

    df_ranking.to_csv("C:/Code/src/Shapely_for_TDA/df_rank.csv")
    return df_ranking

# def avg_std(X_train, y, shap_values):
#     feature_names = X_train.columns.values  # list the feature names
#     label_names = (np.unique(np.array(y))).tolist()
#     new_shap = (pd.DataFrame(np.concatenate(shap_values), columns=feature_names)).round(3)
#     lol = len(X_train)
#     data_name = [name for name in label_names for i in range(lol)]
#     new_shap.insert(0, 'dataset', data_name)
#     mean_std = new_shap.groupby('dataset').agg(['mean', 'std'], axis=0)
#
#     return mean_std

def main():
    x, y = data_csv(datapath)
    x_train, x_test, y_test, clf, accuracy, cross_validation, roc_score = rf_model(x, y)
    features = vanilla_fi(x, x_test, y_test, clf)
    shap_values, class_names = shap_vals(clf, x_train, features)
    df_ranking = shap_feature_ranking(x_train, shap_values, features, class_names)
    #mean_std = avg_std(X_train, y, shap_values)

if __name__ == '__main__':
    datapath = sys.argv[1]
    main()


