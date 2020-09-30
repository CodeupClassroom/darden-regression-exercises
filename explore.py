import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variable_pairs(df, drop_scaled_columns=True):
    if drop_scaled_columns:
        # drop the columns that have _scaled in their name
        scaled_columns = [c for c in df.columns if c.endswith('_scaled')]
        df = df.drop(columns=scaled_columns)
    
    g = sns.PairGrid(df)
    g.map_offdiag(sns.regplot)
    g.map_diag(plt.hist)
    
    return g

def months_to_years(df):
    return df.assign(tenure_years=(df.tenure / 12).apply(math.floor))

def plot_categorical_and_continuous_vars(df, categorical_var, continuous_var):
    sns.barplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.violinplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.boxplot(data=df, y=continuous_var, x=categorical_var)