# Homework4 :telephone:
## Ensembles

For **Homework4** we tried to implement machine learning algorithms, such as **Decision Tree**, **Random Forest**, **CatBOOST** and many others, using custom fuctions for RandomForest Classification, and from `sklearn`, `xgboost`, `lightgbm`, `catboost` Python packages. This task were coded with basic Python and `pandas` package. For data visualization we used `matplotlib` and `seaborn` packages.

During this Homework we were dealing with:

- DecisionTreeRegressor - to demonstrate bias-variance trade-off on simply generated dataset for 500 observations and 1 feature;
- Custom RandomForestClassifier - dataset generated with the `sklearn.dataset.make_moons()` for 500 observations and 2 features;
- RandomForestClassifier - the `sklearn` breast cancer dataset to visualize the `max_features` and `max_depth` parameters influence on base models correlation;
- VotingClassifier - to build and evaluate ensemble with number of models for diagnose heart disease in people by medical indicators ([dataset](./data/heart.csv));
- RandomForest and boosting algorithms - to determine the outflow of customers from telecom using [this dataset](./data/churn.csv)


For models quality evaluation, we used classification metrics, such as *precision*, *recall*, *f1_score*, *accuracy*, and build *confusion matrix* and *ROC-AUC curve*.

### Files

There are **two files** in this folder. Here some discriptions of them.

- [README.md](./README.md): discriptions for files in this directory;

- [requirements.txt](./requirements.txt): .txt file with the dependencies;

### Folders

- [source](./source) folder contains [Ensembles.ipynb Notebook](./source/Ensembles.ipynb) with some solutions for Homework4;

- [data](./data) folder with dataset on [heart disease](./data/heart.csv) and dataset for [telecome](./data/churn.csv) customers outflow prediction

### System

This Homework was prepared on *Ubuntu 22.04.1 LTS* with *Python version 3.10.6*
