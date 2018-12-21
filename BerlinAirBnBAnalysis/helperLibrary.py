import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gmaps
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import helperLibrary as h

visualize = False
debugdf = False

def colinspection(df):
    """
    Prints the first few lines of every columns, so that one can get an insight.
    :param df: takes into a dataframe
    :return:
    """
    columns = df.columns.tolist()
    for col in columns:
        print(df[col].head())
#         print(5*'//')
#         print('Percentage of Missing data:', df[col].isnull().mean())
        print(10*'----')

def checkmissing(df):
    """

    :param df: takes into a dataframe
    :return:
    """
    columns = df.columns.tolist()
    missing = []
    for col in columns:
        misisng = df[col].isnull().mean()

def showmissing(df, lower_thresh=0.0):
    """
    Shows a barplot of the missing values of all columns.
    :param df: takes into a pandas dataframe.
    :param lower_thresh: Allows to threshold the count of missing columns to gain greater insight
    :return:
    """
    plt.figure(figsize=(16,10))
    na_count = df.isna().sum()/df.shape[0]
    na_count = na_count[na_count >= lower_thresh]
    plot = sns.barplot(na_count.index.values, na_count)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=90)


def showdistribution_advanced(df, column=str, lower_thresh=0, upper_thresh=9999999, percent=True, group=False,
                              groupbin=None):
    """

    :param df: takes in a pandas dataframe
    :param column: takes in a string of the dataframe as string
    :param lower_thresh: Allows to specify a lower threshold that cuts the counts using a lower threshold
    :param upper_thresh: Allows to specify a lower threshold that cuts the counts using a upper threshold
    :param percent: boolean value to allow a display of percentage distribution instead of actual counts
    :param group: boolean value to allow in combination with the parameter groupbin to group the data and show the
                    groupcount
    :param groupbin: needs to be specified as a list, an example would be [0,1,2,10] which creates three bins.
    :return: returns nothing, just plots a barplot
    """
    if group:
        if groupbin == None:
            print('Oh no, you have forgotten to specify a bin which should be used.\n'
                  'An example would be: groupbin = [0, 1, 2, 3, 4, 100]')
            pass
        else:
            bins = pd.cut(df[column], groupbin)
            if percent:
                status_vals = df.groupby(bins)[column].agg(['count']) / df.shape[0]
            else:
                status_vals = df.groupby(bins)[column].agg(['count'])
            title = "Grouped count of the listings for the column: " + column
            plot = status_vals.plot.bar(title=title, figsize=(16, 10), rot=1)

    else:
        plt.figure(figsize=(16, 10))
        status_vals = df[column].value_counts()
        status_vals = status_vals[status_vals >= lower_thresh]
        status_vals = status_vals[status_vals <= upper_thresh]
        if percent:
            plt.title("Distribution count of the listings for the column: " + column)
            plot = sns.barplot(status_vals.index.values, status_vals / df.shape[0])
        else:
            plt.title("Value count of the listings for the column: " + column)
            plot = sns.barplot(status_vals.index.values, status_vals)
            plt.grid(True)
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)


def splitdf(df, output=True, full_na=True):
    """
    splits a daataframe into distinct dataframes with zero NaN values and a dataframe with some NaN values
    :param df: takes in a dataframe
    :param output: boolean value that outputs the shape composition of the new dataframes
    :param full_na: boolean value that outputs a list of columns, that has only NaN values
    :return: nothing
    """
    # Generating a subdataframe with zero missing values
    zero_na_df = df[df.columns[df.isnull().mean() == 0]]
    # Generating a subdataframe that has missing values, but still contain valid values
    some_na_df = df[df.columns[(df.isnull().mean() != 0) &
                               (df.isnull().mean() != 1)]]
    # Generating subdataframe with only missing values
    full_na_df = df[df.columns[df.isnull().mean() == 1]]

    if output:
        print(5 * '/', "Shapes of the datasets", 5 * '\\', '\n')
        print('The first dataset has the following shape: ', zero_na_df.shape)
        print('The second dataset has the following shape: ', some_na_df.shape)
        print('The third dataset has the following shape: ', full_na_df.shape)
        print('\n', 10 * '-----', '\n')

    if full_na_df.shape[1] != 0 and full_na:
        print('The following columns should be dropped and not considered further' +
              'as those have only missing values as content\n')
        print(full_na_df.columns.tolist())
    else:
        print('This dataset has no columns with full nan data')

    return zero_na_df, some_na_df