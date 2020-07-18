import numpy as np
from numpy import load
from numpy import save
import sklearn
import random 
import pdb
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import random

def evalSamplingp(ytest,ypred):
    return precision_score(ytest,ypred)
def evalSamplingr(ytest,ypred):
    return recall_score(ytest,ypred)



prediction_y = torch.load("../Code_github/datasets_y_prediction.pt")#, map_location='cpu')#, allow_pickle = True)

y_test_datasets_resampled = np.load("../Code_github/datasets_y_test_resampled.npy", allow_pickle = True)



matrixp =  np.empty((len(y_test_datasets_resampled)*14,21))
matrixr =  np.empty((len(y_test_datasets_resampled)*14,21))



c  = 0
for i in range(len(y_test_datasets_resampled)):

    
    for j in range(21):
        ytest = y_test_datasets_5d_resampled[(i)*21+j]
        c = c + 1
        
        
        
        
        for k in range(14):
            ypred = (prediction_y[(i)*21*14 + k + 14*(j)].data).cpu().numpy()


            precision = evalSamplingp(ytest, ypred)

            matrixp[(i)*14 + k ][j] = round(precision,3)


            recall = evalSamplingr(ytest, ypred)

            matrixr[(i)*14 + k ][j] = round(recall,3)




np.savetxt('../Code_github/datasets_precision.csv', matrixp.tolist() ,delimiter=',',fmt='%f') 
np.savetxt('../Code_github/datasets_recall.csv', matrixr.tolist() ,delimiter=',',fmt='%f') 

