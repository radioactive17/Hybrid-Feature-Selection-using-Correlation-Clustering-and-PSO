import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import math
import time
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


import concurrent.futures

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


if __name__ == '__main__':
    start = time.perf_counter()
    
    df = pd.read_csv('cleaned_arythmia.csv')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = X.drop(columns = ['Unnamed: 0'], axis = 1)

    scaler = MinMaxScaler()
    X_norm = scaler.fit(X)
    X_norm = scaler.transform(X)
    X_norm_df = pd.DataFrame(X_norm)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

    F_strong = list()
    score_list =[]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(symmetric_uncertainty, X_norm_df[col].tolist(), y.tolist()) for col in X_norm_df.columns]
        for f in concurrent.futures.as_completed(results):
            print(f.result())
    # for itr, col in enumerate(X_norm_df.columns):
    #     score = symmetric_uncertainty(X_norm_df[col].tolist(), y.tolist())
    #     F_strong.append((col, score))
    #     score_list.append(score)
    # before = len(F_strong)
    # print(f"Number of features currently = {before}")

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
