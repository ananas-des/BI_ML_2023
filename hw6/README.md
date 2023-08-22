# Homework6 :izakaya_lantern:

## Multiplayer perceptron

For **Homework5** we tried to build **MLP** (multiplayer perceptron) with three multiconnected layers and different activation functions and optimizators using `torch` Python package. For data visualization we used `matplotlib` packages.

During this **Homework** we were dealing with:

- multiclass classification - on [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) dataset with pictures for handwritten hieroglyphs from ten classes;
- the data normalization impact on model;
- the effect of activation functions on the learning rate and accuracy of model predictions using `Sigmoid`, `Tanh`, `GELU`, or `ReLU` activators;
- the influence of optimizers on the learning rate and accuracy of model predictions using `Adam`, `AdamW`, `RMSprop`, or `Adagrad`;
- custom `ReLU` function

The quality of the obtained models was evaluated using `accuracy score`.

### Files

There are **two files** in this folder. Here some discriptions of them.

- [README.md](./README.md): discriptions for files in this directory;

- [requirements.txt](./requirements.txt): .txt file with the dependencies;

### Folders

There is [FC_NN.ipynb Notebook](./source/FC_NN.ipynb) in [source](./source) folder contains  with some solutions for Homework6.

### System

This Homework was prepared on *Ubuntu 22.04.1 LTS* with *Python version 3.10.6*
