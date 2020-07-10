from sklearn.neural_network import MLPRegressor
import numpy as np
import sklearn
import random 
import pdb
import pandas as pd
from sklearn import metrics
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from numpy import load
from numpy import save
#import h5py
#import pickle
import sklearn
import random 
import pdb
from sklearn.metrics import *

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification

import pandas as pd
from imblearn.metrics import geometric_mean_score
from numpy.random import permutation
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_regression


def evalSampling(sampler, classifier, Xtrain, Xtest,ytrain, ytest):
    """Evaluate a sampling method with a given classifier and dataset
    
    Keyword arguments:
    sampler -- the sampling method to employ. None for no sampling
    classifer -- the classifier to use after sampling
    train -- (X, y) for training
    test -- (Xt, yt) for testing
    
    Returns:
    A tuple containing precision, recall, f1 score, AUC of ROC, Cohen's Kappa score, and 
    geometric mean score.
    """
    X = Xtrain
    y = ytrain
    Xt = Xtest
    yt = ytest
    
    if sampler is not None:
        X_resampled, y_resampled = sampler.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
    else:
        classifier.fit(X, y)
        
    yp = classifier.predict(Xt)
    yProb = classifier.predict_proba(Xt)[:,1] # Indicating class value 1 (not 0)

    precision = precision_score(yt, yp)
    recall    = recall_score(yt, yp)

    return (precision, recall, _, _, _, _)


def extractStandardMeasures(X, y):
    """Extracts the standard measures of the given dataset
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    A tuple (N, d, c) where
    N -- no. of samples
    d -- no. of features
    C -- no. of classes
    """
    (N, d) = X.shape
    c = len(set(y))
    
    return (N, d, c)


import sys

class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    # A utility function to print the constructed MST stored in parent[] 
    def printMST(self, parent=None):
        if parent is None:
            parent = self.root
        print("Edge \tWeight")
        for i in range(1,self.V): 
            print(parent[i],"-",i,"\t",self.graph[i][ parent[i] ])
  
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minKey(self, key, mstSet): 
  
        # Initilaize min value 
        min = sys.maxsize 
  
        for v in range(self.V): 
            if key[v] < min and mstSet[v] == False: 
                min = key[v] 
                min_index = v 
  
        return min_index 
  
    # Function to construct MST for a graph  
    # represented using adjacency matrix representation 
    def primMST(self): 
  
        #Key values used to pick minimum weight edge in cut 
        key = [sys.maxsize] * self.V 
        parent = [None] * self.V # Array to store constructed MST 
        # Make key 0 so that this vertex is picked as first vertex 
        key[0] = 0 
        mstSet = [False] * self.V 
  
        parent[0] = -1 # First node is always the root of 
  
        for cout in range(self.V): 
  
            # Pick the minimum distance vertex from  
            # the set of vertices not yet processed.  
            # u is always equal to src in first iteration 
            u = self.minKey(key, mstSet) 
  
            # Put the minimum distance vertex in  
            # the shortest path tree 
            mstSet[u] = True
  
            # Update dist value of the adjacent vertices  
            # of the picked vertex only if the current  
            # distance is greater than new distance and 
            # the vertex in not in the shotest path tree 
            for v in range(self.V): 
                # graph[u][v] is non zero only for adjacent vertices of m 
                # mstSet[v] is false for vertices not yet included in MST 
                # Update the key only if graph[u][v] is smaller than key[v] 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
                        key[v] = self.graph[u][v] 
                        parent[v] = u 
        
        self.root=parent
#         self.printMST(parent) 

g = Graph(5) 
g.graph = [ [0, 2, 0, 6, 0], 
            [2, 0, 3, 8, 5], 
            [0, 3, 0, 0, 7], 
            [6, 8, 0, 0, 9], 
            [0, 5, 7, 9, 0]] 
  
g.primMST(); 


def extractDecisionBoundaryMeasures(X, y, seed=0):
    """Extract the decision boundary measures of the given dataset
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    A tuple (linSep, decBoundCompVar, decBoundComp) where
    linSep -- Linear separability
    # decBoundCompVar -- Variation in decision boundary complexity # n/a for now
    decBoundComp -- Complexity of the decition boundary
    hyperCentres -- Centres of the hyperspheres along with the last column as the corresponding class
    """
    
    # Linear separability: 10-fold statified cross validation error rate using LDA-Bayes classifier 
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=seed)
    lda = LinearDiscriminantAnalysis()
    np.random.seed(seed)
    errors = []
    for trainIdx, testIdx in sss.split(X, y):
        Xs, ys = X[trainIdx], y[trainIdx] # Sample set
        Xt, yt = X[testIdx], y[testIdx] # Test set
        
        lda.fit(Xs, ys)
        pred = lda.predict(Xt)
        
        errors.append(1 - f1_score(yt, pred)) # Error = 1 - F1_score
    
    linSep = np.mean(errors)
    
    # Complexity of decision boundary: compute hyperspheres
    pool = list(range(X.shape[0]))
    size = X.shape
    np.random.shuffle(pool)
    dist = metrics.pairwise.euclidean_distances(X)
    
    hyperCentres = []
    while len(pool) > 0:
        hs = [pool[0]]          # Initialize hypersphere
        centre = X[pool[0]]     # and its centre
        hsClass = y[pool[0]]          # Class of this hypersphere 
        pool.remove(pool[0])    # Remove the initial point from the pool
        mostDistantPair = None
        
        while True and len(pool)>0:
            dist = np.sqrt(np.sum((X[pool] - centre)**2, axis=1))
            nn = pool[np.argmin(dist)]  # Nearest neighbour index
            if y[nn] != hsClass:        # If this point belongs to a different class
                break                   # conclude the set of points in this sphere
            hs.append(nn)               # Otherwise add it to the sphere
            pool.remove(nn)             # and remove it from the pool
            
            centre = np.mean(X[hs], axis=0)
                
        hyperCentres.append(list(centre)+[hsClass])
    
    # Produce a MST using the centres of the hyperspheres as nodes
    hyperCentres = np.array(hyperCentres)

    
    g = Graph(hyperCentres.shape[0])
    g.graph = metrics.pairwise.euclidean_distances(hyperCentres[:,:2])
    g.primMST()
    # Find the number of inter-class edges in the MST
    idx1 = list(range(1,hyperCentres.shape[0]))
    idx2 = g.root[1:]
    
    N_inter = sum(hyperCentres[idx1,size[1]] != hyperCentres[idx2,size[1]])
    decBoundComp = N_inter / hyperCentres.shape[0] 
    
    return (linSep, decBoundComp, N_inter,hyperCentres,idx1,idx2)



from scipy.special import gamma

def extractTopologyMeasures(X, y):
    """Extract the topology measures of the given dataset
    
    Keyword arguments:
    X -- Design matrix (feature set)
    y -- Label vector
    
    Returns:
    A tuple (groupsPerClass, sdVar, scaleVar) where
    samplesPerGroup
    groupsPerClass
    sdVar -- Variation in feature standard deviation
    # scaleVar -- Scale variation [Not yet implemented]
    """
    
    (linSep, decBoundComp, _,hyperCentres,_,_) = extractDecisionBoundaryMeasures(X, y)
    
    samplesPerGroup = X.shape[0]/hyperCentres.shape[0] 
    classes = list(set(hyperCentres[:,hyperCentres.shape[1]-1]))
    groupsPerClass = [sum(hyperCentres[:,hyperCentres.shape[1]-1]==c) for c in classes]
    sdVar = [np.std(np.std(X[y==c], axis=1)) for c in classes]

    return (samplesPerGroup, groupsPerClass, sdVar)


def volume_overlap(X,y):
    r,c = np.shape(np.array(X))
    voverlap = 1.0
    for i in range(c):
        
        max0 = -100000.0
        max1 = -100000.0
        min0 = 100000.0
        min1 = 100000.0
        
        for j in range(r):
            if(y[j] == 0):
                if(max0 < X[j,i]):
                    max0 = X[j,i]
                    
                if(min0 > X[j,i]):
                    min0 = X[j,i]
                    
                    
            if(y[j] == 1):
                if(max1 < X[j,i]):
                    max1 = X[j,i]
                    
                if(min1 > X[j,i]):
                    min1 = X[j,i]
                    
                    
        voverlap = voverlap * ((np.minimum(max0,max1) - np.maximum(min0,min1))/ (np.maximum(max0,max1) - np.minimum(min0,min1)))
        
        
    return voverlap
 
    
X_train_datasets = load('../Code_github/datasets_X_train_unsampled.npy',allow_pickle = True)
y_train_datasets = load('../Code_github/datasets_y_train_unsampled.npy',allow_pickle = True)


matrix = np.empty((14*len(X_train_datasets),8))

threshold = np.arange(0.2, 0.851, 0.05).tolist() ## 0.2 to 0.85 at an interval of 0.05
counter = 0

for j in range(0,14*len(X_train_datasets),14):

    distance = []
    
    linSep, decBoundComp, N_inter,hyperCentres,idx1,idx2 = extractDecisionBoundaryMeasures(X_train_datasets[counter], np.squeeze(y_train_datasets[counter]), seed=0)
    size = hyperCentres.shape
    for i in range(len(idx1)):
        if(hyperCentres[idx1[i],size[1]-1] != hyperCentres[idx2[i],size[1]-1]):
            distance.append(np.linalg.norm(hyperCentres[idx1[i],0:2] - hyperCentres[idx2[i],0:2]))
    samplesPerGroup, groupsPerClass, sdVar = extractTopologyMeasures(X_train_datasets[counter], np.squeeze(y_train_datasets[counter]))
    voverlap = volume_overlap(X_train_datasets[counter], y_train_datasets[counter])
    

    
    c = 0
        
    for i in range(j,j+14):
        matrix[i][0] = round(linSep,3)
        matrix[i][1] = round(decBoundComp,3)
        matrix[i][2] = round(hyperCentres.shape[0],3)
        matrix[i][3] = round(samplesPerGroup,3)
        matrix[i][4] = round(N_inter,3)
        matrix[i][5] = round(np.mean(distance),3)
        matrix[i][6] = round(voverlap,3)
        matrix[i][7] = round(threshold[c],3)
        c = c+1
        j = j+1
        
    counter = counter +1


np.savetxt('../Code_github/data_metrics.csv', matrix.tolist() ,delimiter=',',fmt='%f')

    
