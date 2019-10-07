"""
Helper function to make use of twitter's analytics
Takes in a twitter analytics csv
Saves a plot
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

source = os.path.join('ignore', 'eda', 'twitter_an', 'tweet.csv')
target = os.path.join('ignore', 'eda', 'twitter_an', 'tweets.png')

def make_plot(source=source, stats='impressions', savepath=target):
    df = pd.read_csv('twitter_an/tweet.csv')
    df['day'] = df.time.apply(lambda x: x[5:10])
    g = df.groupby(df.day).sum()
    plt.style.use("dark_background")
    plt.figure(figsize=(16,16),dpi=128)
    sns.barplot(x=g.index, y=g[stats])
    plt.xlabel('')
    plt.ylabel('')
    
    # plt.ylabel('ylabel', fontsize=24)
    # plt.xlabel('xlabel', fontsize=24, )
    # plt.xlim(min,max)
    # plt.ylim(min,max)
    # plt.title('title')

    sns.despine()
    plt.savefig(savepath)

    return None