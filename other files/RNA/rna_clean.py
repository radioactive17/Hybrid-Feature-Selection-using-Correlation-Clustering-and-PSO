import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

df = pd.read_csv('rna.csv')

df = df.drop(['Unnamed: 0'], axis = 1)
print(df.head())