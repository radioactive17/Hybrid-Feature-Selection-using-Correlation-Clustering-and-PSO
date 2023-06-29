import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import random
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    def __init__(self, X, y, swarm_size, iteration):
        self.X = X
        self.y = y
        self.swarm_size = swarm_size
        self.iteration = iteration
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X.values, self.y, test_size=0.2)
    
    def entropy(self, values):
        probabilities = [float(values.count(value)) / len(values) for value in set(values)]
        return - sum([p * math.log2(p) for p in probabilities])

    def joint_entropy(self, X, Y):
        XY = list(zip(X, Y))
        return entropy(XY)

    def conditional_entropy(self, X, Y):
        return joint_entropy(X, Y) - entropy(Y)

    def symmetric_uncertainty(self, X, Y):
        HX = entropy(X)
        HY = entropy(Y)
        HXY = joint_entropy(X, Y)
        HX_given_Y = conditional_entropy(X, Y)
        return (2 * HX - HX_given_Y) / (HX + HY)
    
    def calculate_crelevance(self):    
        F_strong = list()
        score_list =[]
        for itr, col in enumerate(self.X.columns):
            score = symmetric_uncertainty(self.X[col].tolist(), self.y.tolist())
            F_strong.append((col, score))
            score_list.append(score)
        return score_list, F_strong
    
    def filter(self, score_list, F_max):
        Su_max = max(score_list)
        #print(f"Su_max = {Su_max}")

        D = len(self.X.columns)
        SU_D = D / math.log2(D)
        SU_D = round(SU_D)
        #print(f"SU_D = {SU_D}, F_score of SU_D = {F_strong[SU_D][1]}")
        rho= min(0.1* Su_max, F_strong[SU_D][1])

        for t in F_strong:
            if t[1] < rho:
                F_strong.remove(t)
        return F_strong
    
    def sort_crelevance(self, F_strong):
        F_strong.sort(key=lambda a: a[1], reverse = True)
        return F_strong
    
    def clustering(self, F_strong):
        U0 = F_strong.copy()
        clusters = dict()
        k = 1
        dt_max = U0[0][1] - U0[-1][1]
        D_star = len(U0)
        rho1 = (dt_max*math.log2(D_star))/D_star
        print(f"rho1 = {rho1}")

        while True:
            U1 = U0.copy()
            clusters[k] = list()
            clusters[k].append(U0[0])
            # print(clusters)

            #removing clusters having weak correlation with the top feature(f1) in U1
            for i in range(1, len(U1)):
                difference = U0[0][1] - U0[i][1]
                if difference > rho1:
                    U1.remove(U0[i])

            #Finding similar features to f1(to-feature) and storing them to the cluster-K
            for i in range(1, len(U1)):
                if symmetric_uncertainty(self.X[U1[0][0]].tolist(), self.X[U1[i][0]].tolist()) >= min(U1[0][1], U1[i][1]):
                    clusters[k].append(U1[i])

            #updating the U0
            for values in clusters[k]:
                U0.remove(values)

            #checking the stopping criteria
            if len(U0) > 1:
                k += 1
            else:
                return clusters
                break
                
    def initialize_particles(self, clusters):
        # Initalizaing the swarm
        # Input: M feature clusters
        # Output: N particles 

        #selected probability for each cluster
        max_list = list()
        for values in clusters.values():
            max_list.append(values[0][1])
        probabilities = list()
        for values in max_list:
            probabilities.append(values/max_list[0])

        M = len(list(clusters.keys()))
        particles = np.zeros((self.swarm_size, M))
        for i in range(self.swarm_size):
            for j in range(M):
                if random.uniform(0, 1) < probabilities[i]:
                    index = random.randint(0, len(clusters[j+1])-1)
                    particles[i][j] = index
                else:
                    particles[i][j] = 0

        return particles
    
    def fitness_function(self, position, clusters):
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        #print(X_train)
        #print(type(self.X_train))
        selected_features = list()
        for itr, pos in enumerate(position):
            selected_features.append(clusters[itr+1][int(pos)][0])
        #print(selected_features)
        X_train_selected = self.X_train[:, selected_features]
        X_test_selected = self.X_test[:, selected_features]
        clf = RandomForestClassifier()
        clf.fit(X_train_selected, self.y_train)
        accuracy = clf.score(X_test_selected, self.y_test)
        return accuracy

    def evolutionary(self, clusters, particles):  
        M = len(clusters)
        Pbest = np.zeros((self.swarm_size, M))
        Gbest = np.zeros((1, M))
        itr = 0
        while itr < self.iteration:
            #print(itr)
            for i in range(self.swarm_size):
                if self.fitness_function(particles[i], clusters) > self.fitness_function(Pbest[i], clusters):
                    Pbest[i] = particles[i]

            for i in range(self.swarm_size):
                if self.fitness_function(Pbest[i], clusters) > self.fitness_function(Gbest[0], clusters):
                    Gbest[0] = Pbest[i]
            #print(Pbest)
            
            #print(Gbest)
            for i in range(self.swarm_size):
                #Calculate pm
                temp_sum = 0
                for j in range(M):
                    if Pbest[i][j] == Gbest[0][j]:
                        temp_sum += 1
                pm = 0.1 * (temp_sum/M)

                #Update
                for j in range(M):
                    sum1 = Pbest[i][j] + Gbest[0][j]
                    first_half = math.ceil(sum1/2)
                    abs_diff = abs(Pbest[i][j] - Gbest[0][j])
                    gauss = abs(np.random.normal(loc = 0, scale = 1))
                    second_half = np.ceil(gauss*abs_diff)
                    final = first_half + second_half
                    #print(final)
                    if random.uniform(0, 1) < pm:
                        if final < len(clusters[j+1]):
                            particles[i][j] = final
                    else:
                        particles[i][j] = Pbest[i][j]
            itr += 1
        return Pbest, Gbest
    
    def final_set(self, Gbest, clusters):
        final_features = list()
        for itr, index in enumerate(Gbest[0]):
            if int(index) != 0:
                final_features.append(clusters[itr+1][int(index)][0])
        return final_features
    
    def calculate_classification_accuracy(self, features):
        X_train_selected = self.X_train[:, features]
        X_test_selected = self.X_test[:, features]
        clf = RandomForestClassifier()
        clf.fit(X_train_selected, y_train)
        accuracy = clf.score(X_test_selected, y_test)
        return accuracy
    

if __name__ == '__main__':
    fs = FeatureSelector(X_norm_df, y, 25, 1)
    score_list, F_strong = fs.calculate_crelevance()
    F_strong = fs.filter(score_list, F_strong)
    F_strong = fs.sort_crelevance(F_strong)
    clusters = fs.clustering(F_strong)
    particles = fs.initialize_particles(clusters)
    Pbest, Gbest = fs.evolutionary(clusters, particles)
    features = fs.final_set(Gbest, clusters)
    
    X_train, X_test, y_train, y_test = train_test_split(X_norm_df.values, y, test_size=0.2)
    X_train_selected = X_train[:, features]
    X_test_selected = X_test[:, features]
    clf = RandomForestClassifier()
    clf.fit(X_train_selected, y_train)
    accuracy = clf.score(X_test_selected, y_test)
    print(accuracy)