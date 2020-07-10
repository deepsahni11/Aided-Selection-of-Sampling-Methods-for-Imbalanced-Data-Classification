

# from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random

data_metrics = np.array(pd.read_csv('../Code_github/data_metrics.csv'), header = None)
X = data_metrics



dataset_recall = np.array(pd.read_csv('../Code_github/datasets_recall.csv'), header = None)
y = dataset_recall


dataset_precision = np.array(pd.read_csv('../Code_github/datasets_precision.csv'), header = None)
yprecision = dataset_precision






X_train_datasets_1_bootstrap = []
y_train_datasets_1_bootstrap = []
X_test_datasets_1_bootstrap = []
y_test_datasets_1_bootstrap = []


l = len(X)

def split_random(matrix, percent_train, percent_test, seed):
    """
    Splits matrix data into randomly ordered sets 
    grouped by provided percentages.

    
    """

    np.random.seed = seed
#     percent_validation = 100 - percent_train - percent_test

#     if percent_validation < 0:
#         print("Make sure that the provided sum of " + \
#         "training and testing percentages is equal, " + \
#         "or less than 100%.")
#         percent_validation = 0
#     else:
#         print("percent_validation", percent_validation)

    #print(matrix)  
    rows = matrix.shape[0]
    np.random.shuffle(matrix)

    end_training = int(rows*percent_train/100)    
    end_testing = end_training + int((rows * percent_test/100))

    training = matrix[:end_training]
    testing = matrix[end_training:end_testing]
    #validation = matrix[end_testing:]
    return training, testing




trainx, testx = split_random(X, percent_train=90, percent_test=10, seed = 2) 
trainy, testy = split_random(y, percent_train=90, percent_test=10, seed = 2) 
trainyp, testyp = split_random(yprecision, percent_train=90, percent_test=10, seed = 2) 


print("trainingx",trainx.shape)
print("testingx",testx.shape)
print("trainingy",trainy.shape)
print("testingy",testy.shape)






X_sparse_train = coo_matrix(trainx)
X_sparse_test = coo_matrix(testy)


############### RECALL ################################

for i in range(50):
    trainx_resampled, X_sparse_train_resampled, trainy_resampled = resample(trainx, X_sparse_train , trainy, random_state=i)
    Xtrain = trainx_resampled
    ytrain = trainy_resampled

    X_train_datasets_1_bootstrap.append(Xtrain)
    y_train_datasets_1_bootstrap.append(ytrain)



for i in range(50):
    testx_resampled, X_sparse_test_resampled, testy_resampled = resample(testx, X_sparse_test , testy, random_state=i)
    Xtest = testx_resampled
    ytest = testy_resampled


    X_test_datasets_1_bootstrap.append(Xtest)
    y_test_datasets_1_bootstrap.append(ytest)


np.save("../Code_github/X_train_datasets_bootstrap.npy",X_train_datasets_1_bootstrap)
np.save("../Code_github/y_train_recall_datasets_bootstrap.npy",y_train_datasets_1_bootstrap)
np.save("../Code_github/X_test_datasets_bootstrap.npy",X_test_datasets_1_bootstrap)
np.save("../Code_github/y_test_recall_datasets_bootstrap.npy",y_test_datasets_1_bootstrap)


######### PRECISION #############################
    
    
y_train_datasets_1_bootstrap = []
y_test_datasets_1_bootstrap = []
    
    
    
    
for i in range(50):
    trainx_resampled, X_sparse_train_resampled, trainyp_resampled = resample(trainx, X_sparse_train , trainyp, random_state=i)
    ytrain = trainyp_resampled

    y_train_datasets_1_bootstrap.append(ytrain)



for i in range(50):
    testx_resampled, X_sparse_test_resampled, testyp_resampled = resample(testx, X_sparse_test , testyp, random_state=i)
    ytest = testy_resampled

    y_test_datasets_1_bootstrap.append(ytest)


    

np.save("../Code_github/y_train_precision_datasets_bootstrap.npy",y_train_datasets_1_bootstrap)
np.save("../Code_github/y_test_precision_datasets_bootstrap.npy",y_test_datasets_1_bootstrap)











