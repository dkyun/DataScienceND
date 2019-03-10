import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from helperfunctions import *


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


import workspace_utils


def load_datasets():
    """
    This functions shortens the loading of the datasets used for this analysis
    :return: portfolio, profile, transcript dataset in that order
    """
    # read in the json files
    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('data/profile.json', orient='records', lines=True)
    transcript = pd.read_json('data/transcript.json', orient='records', lines=True)
    return portfolio, profile, transcript


def prepareData(filepath, fillnan=True):
    if os.path.exists(filepath):
        print('File exists, loading file')
        df = pd.read_csv(filepath)
        print('File successfully loaded')
        df.drop(columns='Unnamed: 0', inplace=True)
    else:
        portfolio, profile, transcript = load_datasets()

        profile.rename(index=str, columns={"id": "person"}, inplace=True)
        merged = transcript.merge(profile, on='person', how='outer')
        # Those will be dropped, as there are full of NA values
        df = merged.dropna()

        # Reengineering the value column;
        df['amount'] = df[df.event == 'transaction'].value.apply(lambda x: x['amount'])
        df['offer_id'] = df[(df.event != 'transaction') & (df.event != 'offer completed')].value.apply(
            lambda x: x['offer id'])
        df['offer_id'].update(df[df.event == 'offer completed'].value.apply(lambda x: x['offer_id']))
        df.drop(columns='value', inplace=True)

        # Merging the dataframes together
        portfolio.rename(index=str, columns={'id': 'offer_id'}, inplace=True)
        df = df.merge(portfolio, on='offer_id', how='outer')
        df['duration'] = df['duration'].apply(lambda x: x * 24)

        # Encoding the gender column
        df = pd.concat([df, pd.get_dummies(df['gender'], prefix='gender')], axis=1)
        df.drop(['gender'], axis=1, inplace=True)

        # Encoding the Channels column
        onehotenc_channels = pd.get_dummies(df.channels.apply(pd.Series).stack()).sum(level=0)
        df = pd.concat([df, onehotenc_channels], axis=1, join_axes=[df.index])
        df.drop(columns='channels', inplace=True)

        # Engineering the became member on. We want to keep the year and the total number of months
        df['became_member_on'] = pd.to_datetime(df['became_member_on'], format='%Y%m%d')
        df['MemberSince'] = df.became_member_on.dt.year
        df['MembershipInMonths'] = pd.to_datetime('20180926', format='%Y%m%d')
        df['MembershipInMonths'] = round(((df.MembershipInMonths - df.became_member_on) / np.timedelta64(1, 'M')))
        df.drop(columns='became_member_on', inplace=True)

        # Reengineering some other columns
        persons = df.person.unique()
        for i in tqdm(range(len(persons))):
            subdf = df[df.person == persons[i]]
            for i, row in subdf[subdf.event == 'offer received'].iterrows():
                offer_id = row.offer_id
                maxtime = row.duration + row.time
                offer_index = i
                total_spend = []

                if (row.offer_type == ('bogo' or 'discount')):
                    for i, row in subdf[subdf.event != 'offer received'].iterrows():
                        if (row.event == 'offer viewed' and row.offer_id == offer_id and row.time <= maxtime):
                            df.loc[offer_index, 'OfferViewed'] = 1
                            df.loc[offer_index, 'OfferViewedTime'] = row.time
                        if (row.event == 'offer completed' and row.offer_id == offer_id and row.time <= maxtime):
                            df.loc[offer_index, 'OfferCompleted'] = 1
                            df.loc[offer_index, 'OfferCompletedTime'] = row.time
                        if (row.event == 'transaction' and row.time <= maxtime):
                            df.loc[offer_index, 'TransactionInTime'] = 1
                            df.loc[offer_index, 'TransactionTime'] = row.time
                            total_spend.append(row.amount)

                else:
                    for i, row in subdf[subdf.event != 'offer received'].iterrows():
                        if (row.event == 'offer viewed' and row.offer_id == offer_id and row.time <= maxtime):
                            df.loc[offer_index, 'OfferViewed'] = 1
                            df.loc[offer_index, 'OfferViewedTime'] = row.time
                        if (row.event == 'transaction' and row.time <= maxtime):
                            df.loc[offer_index, 'TransactionInTime'] = 1
                            df.loc[offer_index, 'TransactionTime'] = row.time

                df.loc[offer_index, 'AmountDurProm'] = sum(total_spend)
                df.loc[offer_index, 'NumOfTransactions'] = len(total_spend)

        # Encoding the Offer type
        dummy = pd.get_dummies(df.offer_type)
        df = pd.concat([df, dummy], axis=1)
        df.drop(columns='offer_type', inplace=True)

        # Saving newly created
        df.to_csv(filepath)

        if fillnan:
            df.fillna(0, inplace=True)
    return df


def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    # instantiate kmeans
    kmeans = KMeans(n_clusters=center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))

    return score


def plot_kmeans_knee(df, centerrange=15):
    scores = []
    centers = list(range(1, centerrange))

    for center in centers:
        scores.append(get_kmeans_score(df, center))

    plt.plot(centers, scores, linestyle='--', marker='o', color='b');
    plt.xlabel('K');
    plt.ylabel('SSE');
    plt.title('SSE vs. K');


# Function taken from Lesson 4, chapter 13, DSND Course
def scree_plot(pca, save=1, title='ExplainedVariance.png'):
    '''
    Creates a scree plot associated with the principal components

    INPUT: pca - the result of instantian of PCA in scikit learn

    OUTPUT:
            None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    f = plt.figure(figsize=(16, 10))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i] * 100)[:4])), (ind[i] + 0.2, vals[i]), va="bottom", ha="center",
                    fontsize=12)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')

    if save:
        f.savefig(title)


def do_pca(n_components, data, random_st=42):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    INPUT: n_components - int - the number of principal components to create
           data - the data you would like to transform

    OUTPUT: pca - the pca object created after fitting the data
            X_pca - the transformed X matrix with new number of components
    '''
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components, random_state=random_st)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


# Function taken from Mini-Project Solution
def pca_results(full_dataset, pca, save=1, start=0, finish=1337):
    """
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    :param save: boolean flag, 1, if image shall be saved
    :param full_dataset: dataframe, that is used to display data
    :param pca: The pca data
    :param start: THe begining parameter of which pca components should be displayed
    :param finish: The end number, of which pca components should be displayed
    :return:
    """
    if finish == 1337:
        finish = len(pca.components_)

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)][start:finish]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())[start:finish]
    # 	components = components
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])[start:finish]
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_[start:finish]):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    if save:
        fig.savefig('PcaResult.png')

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


def fit_kmeans(data, centers, random_st=42):
    '''
    INPUT:
        data = the dataset you would like to fit kmeans to (dataframe)
        centers = the number of centroids (int)
    OUTPUT:
        labels - the labels for each datapoint to which group it belongs (nparray)

    '''
    kmeans = KMeans(centers, random_state=random_st)
    labels = kmeans.fit_predict(data)
    return labels