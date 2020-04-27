import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class vizdata:
    name = None 
    def __init__(self):
        self.name = "Data Vizualization for Data Frame"
    
    @staticmethod
    def categorical_features(df, x=None, y=None, hue=None, palette=sns.color_palette("Set2"), ax=None, order=None, verbose=True):
        '''
        Helper function that gives a quick summary of a given column of categorical data
        Arguments
        =========
        dataframe: pandas dataframe
        x: str. horizontal axis to plot the labels of categorical data, y would be the count
        y: str. vertical axis to plot the labels of categorical data, x would be the count
        hue: str. if you want to compare it another variable (usually the target variable)
        palette: array-like. Colour of the plot
        Returns
        =======
        Quick Stats of the data and also the count plot
        '''
        if x == None:
            column_interested = y
        else:
            column_interested = x
        series = df[column_interested]
        print(series.describe())
        print('mode: ', series.mode())
        if verbose:
            print('='*80)
            print(series.value_counts())
        sns.countplot(x=x, y=y, hue=hue, data=df,palette=palette, order=order, ax=ax)
        plt.show()
        
    @staticmethod
    def continuous_features(df, x=None, y=None, hue=None, palette=sns.color_palette("Set2"), ax=None, order=None, verbose=True, swarm=False):
        '''
        Helper function that gives a quick summary of quantattive data
        Arguments
        =========
        dataframe: pandas dataframe
        x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
        y: str. vertical axis to plot the quantitative data
        hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
        palette: array-like. Colour of the plot
        swarm: if swarm is set to True, a swarm plot would be overlayed
        Returns
        =======
        Quick Stats of the data and also the box plot of the distribution
        '''
        series = df[y]
        print(series.describe())
        print('mode: ', series.mode())
        if verbose:
            print('='*80)
            print(series.value_counts())
        sns.boxplot(x=x, y=y, hue=hue, data=df,palette=palette, order=order, ax=ax)
        if swarm:
            sns.swarmplot(x=x, y=y, hue=hue, data=df,palette=palette,order=order, ax=ax)
        plt.show()

    @staticmethod
    def corr_features(df):
        '''
        Helper function that gives a quick summary of correlation betwwen the features Arguments
        =========
        df: pandas dataframe
        Returns
        =======
        '''
        plt.figure(figsize=(16,14))
        plt.title('Pearson Correlation of Features', size = 15)
        colormap = sns.diverging_palette(10, 220, as_cmap = True)
        sns.heatmap(df.corr(),cmap = colormap,square = True,annot = True,linewidths=0.1,vmax=1.0, linecolor='white',annot_kws={'fontsize':12})
        plt.show()

    @staticmethod
    def cluster_plot(df,K_min,K_max):
        '''
        Helper function that gives a quick summary of K-Means clustering Elbow plotArguments
        =========
        df: pandas dataframe
        K_min: int. minimum number of clusters
        K_max: int. maximum number of clusters
        Returns
        =======
        Quick plot of the clusters possible
        '''
        scaler = preprocessing.MinMaxScaler()
        NC = range(K_min,K_max)
        kmeans = [KMeans(n_clusters=i) for i in NC]
        score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
        plt.plot(NC,score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Curve')
        plt.show()

    @staticmethod
    def pair_plot_continuous(df,x,y,hue):
        '''
        Helper function that gives a quick summary of pair of variable Arguments
        =========
        df: pandas dataframe
        x: str. first variable in the pair plot
        y: str. second variable in the pair plot
        hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
        Returns
        =======
        Quick pair plot for given variable pair
        '''
        sns.pairplot(df, x_vars=[x], y_vars=[y], height=10, hue=hue, palette=sns.color_palette("Set2"),plot_kws=dict(edgecolor="k", linewidth=0.5))
        plt.show()

    @staticmethod
    def hist_features(df):
        '''
        Helper function that gives a quick summary of feature Arguments
        =========
        df: pandas dataframe
        Returns
        =======
        Quick histogram for given features of data frame
        '''
        df.hist(bins=10, color='steelblue', edgecolor='navy', linewidth=1.0, xlabelsize=5, ylabelsize=5, grid=False)    
        plt.tight_layout(rect=(0,0,1.0,1.0))
        plt.show()
