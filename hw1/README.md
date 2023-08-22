# Homework1 :tshirt:

## *k*-Nearest Neighbors Method

For **Homework1** we tried to implement ***k*-Nearest Neighbors** machine learning algorithm using custom fuctions for classification and model metrics, and comparing their performance with respective functions from `sklearn` Python package. This task were coded with basic Python and `pandas` package. For data visualization we used `matplotlib` and `seaborn` packages.

During this Homework we were dealing with:

- binary classification - on [Fasion MNIST](https://www.kaggle.com/c/cuny-csi-fashion-mnist) dataset and two classes; 
- multiclass classification - on the same dataset with cloth pictures and ten their classes; 
- regression (when the dependent variable is a natural number) - dataset on diabetes from `sklearn` package

Since the method needs a hyperparameter, the number of neighbors (*k*), we coded functions for their selection based on model metrics that are used in *classification* (precision, recall, f1, and accuracy) and *regression* ($R^2$, MSE, and MAE).

### Files

There are **two files** in this folder. Here some discriptions of them.

- [README.md](./README.md): discriptions for files in this directory;

- [requirements.txt](./requirements.txt): .txt file with the dependencies;

### Folders

[source](./source) folder contains three files:

- [KNN.ipynb notebook](./source/KNN.ipynb) with some solutions for Homework1; 
- [knn.py](./source/knn.py) with class *KNNClassifier* with methods:

    - `predict()`: to predict classes for the data samples;
    - `compute_distances_two_loops()`, `compute_distances_one_loop()`, `compute_distances_no_loops()`: to compute L1 distance from each test sample to each training sample with denoted number of loops in code;
    - `predict_labels_binary()`: to predict model for binary classification;
    - `predict_labels_multiclass()`: to predict model for multoclass classification

- [metrics.py](./source/metrics.py) with following functions:
    
    - `binary_classification_metrics()`: to compute metrics for binary classification (precision, recall, f1, and accuracy scores);
    - `multiclass_accuracy()`: to compute accuracy metric for multiclass classification;
    - `r_squared()`: to compute $R^2$ for regression;
    - `mse()`: to compute mean squared error;
    - `mae()`: to compute absolute error
    
### System

This Homework was prepared on *Ubuntu 22.04.1 LTS* with *Python version 3.10.6*
