# Homework7 :herb:

## Convolutional neural networks (CNN)

For **Homework7** we created custom **convolution function** and tried to reproduce **AlexNet** architecture with *five* convolutional layers and a 2D max pooling for feature extraction. For classification stage, we used two linear fully connected (FC) layers. `ReLU` activation function was used for both purposes. Different convolution, activation and optimization was performed using `pytorch` Python package. For data visualization we used `matplotlib` and `seaborn` packages.

During this **Homework** we were dealing with:

- creating custom function for convoluting images with any mask;
- multiclass classification - on [dataset](./data) with pictures for bean leaves of three classes: healthy, with angular leaf spot, or bean rust

The quality of the obtained models was evaluated using `accuracy score`.

### Files

There are **two files** in this folder. Here some discriptions of them.

- [README.md](./README.md): discriptions for files in this directory;

- [requirements.txt](./requirements.txt): .txt file with the dependencies;

### Folder

- [source](./source) folder contains [hw_cnn.ipynb](./source/hw_cnn.ipynb) with some solutions for **Homework7**

Data with **bean leaves** images for model [training](https://storage.googleapis.com/ibeans/train.zip), [validation](https://storage.googleapis.com/ibeans/validation.zip) and [testing](https://storage.googleapis.com/ibeans/test.zip) target prediction could be downloaded via respective link.

### System

This Homework was prepared on *Ubuntu 22.04.1 LTS* with *Python version 3.10.6*
