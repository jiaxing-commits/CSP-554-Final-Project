''' Explore PySpark MLlib Models Metrics '''

import pandas as pd 
import seaborn as sns 
sns.set(style='whitegrid')


def multiple_bar_plot(tool, dataset, df):
    df = pd.melt(df, id_vars='model', var_name='metric')
    plt = sns.catplot(x='metric', y='value', hue='model', data=df, kind='bar')
    plt.set(title='%s Model Metrics for %s Dataset' %(tool, dataset), ylabel='metric', xlabel='')
    plt.despine(left=True)
    plt.set_xticklabels(rotation=45)
    plt.savefig('%s_%s.png' %(dataset.lower(), tool.lower()))


tools = ['Spark', 'Scikit']
datasets = ['Airline']
for tool in tools:
    for dataset in datasets:
        df = pd.read_csv('%s_%s_metric.csv' % (dataset.lower(), tool.lower()))
        multiple_bar_plot(tool, dataset, df)
