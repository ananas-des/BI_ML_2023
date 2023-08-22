import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    
    TP = sum((y_pred == y_true) & (y_pred == 1))
    TN = sum((y_pred == y_true) & (y_pred == 0))
    FP = sum((y_pred != y_true) & (y_pred == 1))
    FN = sum((y_pred != y_true) & (y_pred == 0))
    
    try:
        precision_score = TP / (TP + FP)
    except ZeroDivisionError:
        precision_score = None
        print("All cases have been predicted to be negative, precision score can not be calculated!")
    
    try:
        recall_score = TP / (TP + FN) 
    except ZeroDivisionError:
        recall_score = None
        print("No positive cases in the input data, recall score can not be calculated!")

    try:
        f1_score = 2 * precision_score * recall_score / (precision_score + recall_score)
    except ZeroDivisionError:
        f1_score = None
        print("The classifier cannot predict any correct class!")

    try:
        accuracy_score = (TP + TN) / (TP + TN + FP + FN)
    except ZeroDivisionError:
        accuracy_score = None
        print("The classifier cannot predict any correct class!")
    return precision_score, recall_score, f1_score, accuracy_score
   
    
def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    
    n_y = y_true.shape[0]
    return np.sum((y_pred == y_true)) / n_y


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    
    r2 = 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    
    mse = np.sum(np.square(y_true - y_pred)) / y_true.shape[0]
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    
    mae = np.sum(np.abs(y_true - y_pred)) / y_true.shape[0]
    return mae
    