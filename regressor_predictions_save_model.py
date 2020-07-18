from sklearn.neural_network import MLPRegressor
import numpy as np
import sklearn
import pickle 
import random
import pdb
from sklearn.externals import joblib 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import mean_squared_error
from numpy import load
from numpy import save


training_x = np.load("X_train_real_datasets_bootstrap_sample.npy", allow_pickle = True)[:560] 
test_x = np.load("X_test_real_datasets_bootstrap.npy", allow_pickle = True)

training_y_recall = np.load("y_train_real_recall_datasets_bootstrap_sample.npy", allow_pickle = True)[:560]
training_y_precision = np.load("y_train_real_precision_datasets_bootstrap_sample.npy", allow_pickle = True)[:560]s


file = open('regressor_model.pkl','wb')

for i in range(21):

    c = 1
    for p in [20]: 
        for q in [15]: 

            reg = MLPRegressor(alpha=1e-3,
                               hidden_layer_sizes=(p, q),
                               random_state=1,
                               activation="tanh",
                               batch_size= 64,
                               max_iter=5000)
            
            for b in range(len(training_x)): # number of bootstrapped samples
                
                
                Xtrain = training_x[b]

                ytrainp = training_y_precision[b]
                ytrainr = training_y_recall[b]


                ytrain_p = ytrainp[:,i].reshape(-1,1)
                ytrain_r = ytrainr[:,i].reshape(-1,1)
                ytrain_pr = np.concatenate((ytrain_p,ytrain_r), axis = 1)
        
        
                reg.fit(Xtrain, ytrain_pr)
            
                pickle.dump(reg, file)


                
                c= c+1


file.close()
