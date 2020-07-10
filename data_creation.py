import numpy as np
import sklearn
import random 
import pdb
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import save
from numpy import load

seed = 0

#### Every dataset has been labelled using this sequence 
#### (flip_fraction,num_informative,class_separation,num_clusters,random_seed,num_features,num_classes,
####  num_repeated,num_redundant)


num_datapoints = 1000

flip_fraction = [0,0.002,0.004,0.006,0.008,0.01]
num_informative = [7,8,9]
class_separation = np.arange(0.30, 2.0, 0.35).tolist()
num_clusters = [1,2,3]

random_seed = 0 
num_features = 9
num_classes = 2
num_repeated = 0
num_redundant = 0
weights = [[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4]]

X_train_datasets_unsampled = []
y_train_datasets_unsampled = []
X_test_datasets_unsampled = []
y_test_datasets_unsampled = []

c = 0

for w in weights:
    for f in flip_fraction:
        for num_i in num_informative:
            for cs in class_separation:
                for num_c in num_clusters:
                
                    X,y = make_classification(n_samples=num_datapoints, n_features=num_features, n_informative=num_i, 
                                        n_redundant=num_redundant, n_repeated=num_repeated, n_classes=num_classes, n_clusters_per_class=num_c,
                                           class_sep=cs,
                                       flip_y=f,weights=w, random_state = random_seed)

                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
                    sss.get_n_splits(X, y)
                    for train_index, test_index in sss.split(X, y):
                        Xtrain, Xtest = X[train_index], X[test_index]
                        ytrain, ytest = y[train_index].reshape(-1,1), y[test_index].reshape(-1,1)
                        
                        
                        X_train_datasets_unsampled.append(Xtrain)
                        y_train_datasets_unsampled.append(ytrain)
                        X_test_datasets_unsampled.append(Xtest)
                        y_test_datasets_unsampled.append(ytest)
                           
                    c = c+1
                                         
np.save('../Code_github/datasets_X_train_unsampled.npy',X_train_datasets_unsampled)
np.save('../Code_github/datasets_X_test_unsampled.npy',X_test_datasets_unsampled)
np.save('../Code_github/datasets_y_train_unsampled.npy',y_train_datasets_unsampled)
np.save('../Code_github/datasets_y_test_unsampled.npy',y_test_datasets_unsampled)
