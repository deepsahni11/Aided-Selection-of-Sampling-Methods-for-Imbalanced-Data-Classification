from sklearn.neural_network import MLPRegressor
import numpy as np
import sklearn
import random
import pdb
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import mean_squared_error

from numpy import load
from numpy import save


training_x = np.load("../Code_github/X_train_datasets_bootstrap.npy", allow_pickle = True) 
test_x = np.load("../Code_github/X_test_real_datasets_bootstrap.npy", allow_pickle = True)

training_y = np.load("../Code_github/y_train_precision_datasets_bootstrap.npy", allow_pickle = True)


predictions_precision = []


for b in range(len(training_x)): # number of bootstrapped samples

    Xtrain = training_x[b]
    ytrain = training_y[b]
    Xtest = test_x[b]

    predictions_precision = []

    
   
  
    c = 1
    for p in [20]: 
        for q in [15]: 

            reg = MLPRegressor(alpha=1e-4,
                               hidden_layer_sizes=(p, q),
                               random_state=1,
                               activation="tanh",
                               batch_size= 64,
                               max_iter=500)
            

            for i in range(21):
                predictions = []
                reg.fit(Xtrain, ytrain[:,i])

                for j in range(len(Xtest)):
                    pred_y_test = reg.predict(Xtest[j].reshape(1,-1))
                    predictions.append(pred_y_test)
                    
                prediction = np.array(predictions).reshape(-1,1)
                predictions_precision.append(prediction)

                
                c= c +1



    np.save("../Code_github/precision_predictions_bootstrap_" + str(b) + ".npy" , predictions_precision)
