# EECS 445 - Fall 2019
# Project 1 - helper.py

import pandas as pd
import numpy as np

import project1


def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)


def get_split_binary_data(class_size=500):
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Also returns the dictionary used to create the feature matrices.
    Input:
        class_size: Size of each class (pos/neg) of training dataset.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    X_train = pd.concat([positiveDF[:class_size], negativeDF[:class_size]]).reset_index(drop=True).copy()
    dictionary = project1.extract_dictionary(X_train)
    X_test = pd.concat([positiveDF[class_size:(int(1.5 * class_size))],
                        negativeDF[class_size:(int(1.5 * class_size))]]).reset_index(drop=True).copy()
    Y_train = X_train['label'].values.copy()
    Y_test = X_test['label'].values.copy()
    X_train = project1.generate_feature_matrix(X_train, dictionary)
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test, dictionary)


def get_imbalanced_data(dictionary, positive_class_size=800, ratio=0.25):
    """
    Reads in the data from data/imbalanced.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are binary as follows
        -1: poor/average
        1: good
    Input:
        dictionary: the dictionary created via get_split_binary_data
        positive_class_size: the size of the positive data
        ratio: ratio of negative_class_size to positive_class_size
    """
    fname = "data/imbalanced.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    dataframe = pd.concat([positiveDF[:positive_class_size],
                           negativeDF[:(int(positive_class_size * ratio))]]).reset_index(drop=True).copy()
    X_train = project1.generate_feature_matrix(dataframe, dictionary)
    Y_train = dataframe['label'].values.copy()

    return (X_train, Y_train)


def get_imbalanced_test(dictionary, positive_class_size=200, ratio=0.25):
    """
    Reads in the data from data/dataset.csv and returns a subset of it
    reflecting an imbalanced test dataset
        -1: poor/average
        1: good
    Input:
        dictionary: the dictionary created via get_split_binary_data
        positive_class_size: the size of the positive data
        ratio: ratio of negative_class_size to positive_class_size
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe['label'] != 0]
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    X_test = pd.concat([positiveDF[:positive_class_size],
                        negativeDF[:int(positive_class_size * ratio)]]).reset_index(drop=True).copy()
    Y_test = X_test['label'].values.copy()
    X_test = project1.generate_feature_matrix(X_test, dictionary)

    return (X_test, Y_test)


def get_multiclass_training_data(class_size=400, Count = False, Rating = False, Punc = False):
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    Input:
        class_size: Size of each class (pos/neg/neu) of training dataset.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    neutralDF = dataframe[dataframe['label'] == 0].copy()
    positiveDF = dataframe[dataframe['label'] == 1].copy()
    negativeDF = dataframe[dataframe['label'] == -1].copy()
    X_train = pd.concat([positiveDF[:class_size], negativeDF[:class_size], neutralDF[:class_size]]).reset_index(drop=True).copy()
    dictionary = project1.extract_dictionary(X_train)
    Y_train = X_train['label'].values.copy()
    X_train = project1.generate_feature_matrix_multi(X_train, dictionary, Count, Rating, Punc)

    return (X_train, Y_train, dictionary)


def get_heldout_reviews(dictionary):
    """
    Reads in the data from data/heldout.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix
    Input:
        dictionary: the dictionary created by get_multiclass_training_data
    """
    fname = "data/heldout.csv"
    dataframe = load_data(fname)
    X = project1.generate_feature_matrix(dataframe, dictionary)
    return X


def generate_challenge_labels(y, uniqname):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the ratings in the heldout dataset since we will
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(uniqname+'.csv', header=['label'], index=False)
    return
