import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation(df):
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.show()