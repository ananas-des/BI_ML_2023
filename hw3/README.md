# Homework3 :high_brightness:

## Clustering. tSNE

For **Homework3** we tried to implement **Clustering** machine learning algorithms, such as **KMeans**, **Agglomerative Clustering**, and **DBSCAN**, using custom fuctions for Kmeans, and from `sklearn` Python package. This task were coded with basic Python and `pandas` package. For data visualization we used `matplotlib`, `seaborn`, and `IPython` packages.

During this Homework we were dealing with:

- Kmeans Clustering - simply generated dataset for 400 observations and 2 features;
- Agglomerative Clustering - the `sklearn` digits dataset;
- Kmeans, Agglomerative Clustering, and DBSCAN - dataset on flow cytometry results for [immune cells](./data/flow_c_data.csv) type clusterization;
- kNN and Logical Regression for cell type classification


For models quality evaluation, we used distance metrics and *homogeneity*, *completeness*, *v-measure*, and *silhouette* scores. Also we visualized resulted clustering using tSNE and built confusion matrix for further cell type classification.

### Files

There are **two files** in this folder. Here some discriptions of them.

- [README.md](./README.md): discriptions for files in this directory;

- [requirements.txt](./requirements.txt): .txt file with the dependencies;

### Folders

- [source](./source) folder contains [Cluster_tSNE.ipynb Notebook](./source/Cluster_tSNE.ipynb) with some solutions for Homework3;

- [data](./data) folder with dataset on flow cytometry for [immune cells](./data/flow_c_data.csv) and dataset with [resulted clusterization](./data/labeled_fc_data.csv)

### System

This Homework was prepared on *Ubuntu 22.04.1 LTS* with *Python version 3.10.6*
