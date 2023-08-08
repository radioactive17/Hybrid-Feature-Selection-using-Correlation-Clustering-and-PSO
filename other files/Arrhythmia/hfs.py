import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import math
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool

df = pd.read_csv('cleaned_arythmia.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.drop(columns = ['Unnamed: 0'], axis = 1)
scaler = MinMaxScaler()
X_norm = scaler.fit(X)
X_norm = scaler.transform(X)
X_norm_df = pd.DataFrame(X_norm)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

def entropy(values):
    probabilities = [float(values.count(value)) / len(values) for value in set(values)]
    return - sum([p * math.log2(p) for p in probabilities])

def joint_entropy(X, Y):
    XY = list(zip(X, Y))
    return entropy(XY)

def conditional_entropy(X, Y):
    return joint_entropy(X, Y) - entropy(Y)

def symmetric_uncertainty(X, Y):
    HX = entropy(X)
    HY = entropy(Y)
    HXY = joint_entropy(X, Y)
    HX_given_Y = conditional_entropy(X, Y)
    return (2 * HX - HX_given_Y) / (HX + HY)

def calculate_score(args):
    col, X_col, y = args
    score = symmetric_uncertainty(X_col, y)
    return (col, score)

# Assuming you have imported the symmetric_uncertainty function and defined X_norm_df and y

F_strong = []
score_list = []

# Convert DataFrames to NumPy arrays for faster computations
X_norm_arr = X_norm_df.values
y_arr = y.values

# Prepare arguments for multiprocessing
args_list = [(col, X_norm_arr[:, itr], y_arr) for itr, col in enumerate(X_norm_df.columns)]

# Number of CPU cores to use in parallel processing
num_cores = 4  # You can adjust this based on the number of CPU cores available.

# Use multiprocessing to calculate scores in parallel
with Pool(processes=num_cores) as pool:
    results = pool.map(calculate_score, args_list)

# Collect the results from multiprocessing
F_strong = [(col, score) for col, score in results]
score_list = [score for _, score in results]

before = len(F_strong)
print(f"Number of features currently = {before}")
