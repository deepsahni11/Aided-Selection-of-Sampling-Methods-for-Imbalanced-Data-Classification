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
test_x = np.load("../Code_github/X_test_datasets_bootstrap.npy", allow_pickle = True)

training_y = np.load("../Code_github/y_train_recall_datasets_bootstrap.npy", allow_pickle = True)
test_y = np.load("../Code_github/y_test_recall_datasets_bootstrap.npy", allow_pickle = True)

rs_test = []
rs_train = []
training_loss = []
test_loss = []



rmse_bootstrap = []
mean_rmse = []
hidden_layers = []

for b in range(len(training_x)): # number of bootstrapped samples

    Xtrain = training_x[b]
    ytrain = training_y[b]
    Xtest = test_x[b]
    ytest = test_y[b]

    rs_test = []
    rs_train = []
    training_loss = []
    test_loss = []

    mean_rmse = []

    rmse_sampling = None

    rmse_min = 10

    h1 = None
    h2 = None
    for p in [20]: #[2,3,5,8,10,15,20,25,30,35,40]:
        for q in [15]: #[2,3,5,8,10,15,20,25,30,35,40]:

            rmse = np.empty((21,1))
            reg = MLPRegressor(alpha=1e-4,
                               hidden_layer_sizes=(p, q),
                               random_state=1,
                               activation="tanh",
                               batch_size= 64,
                               max_iter=500)
            predictions = []

            for i in range(21):
                reg.fit(Xtrain, ytrain[:,i])

                pred_y_test = reg.predict(Xtest[0:len(Xtest)-1])
                prediction = pred_y_test.reshape(-1,1)
                rmse_scalar = np.sqrt(mean_squared_error(ytest[:,i].reshape(-1,1)[0:len(Xtest)-1], prediction))
                rmse[i,0] = np.round(rmse_scalar,3)




    mean_rmse.append(np.mean(rmse_sampling))
    rmse_bootstrap.append(rmse_sampling)
    print(rmse_sampling)
    print("rmse mean" , np.mean(rmse_sampling))
    np.save("../Code_github/rmse_sampling_for_recall_bootstrap_" + str(b) + ".npy" , rmse_sampling)
