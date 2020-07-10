import numpy as np
import random 
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from numpy import save
from numpy import load
import timeit


X_train_datasets_resampled = load('../Code_github/datasets_X_train_resampled.npy',allow_pickle = True)
X_test_datasets_resampled = load('../Code_github/datasets_X_test_resampled.npy',allow_pickle = True)
y_train_datasets_resampled = load('../Code_github/datasets_y_train_resampled.npy',allow_pickle= True)
y_test_datasets_resampled = load('../Code_github/datasets_y_test_resampled.npy',allow_pickle = True)


threshold = np.arange(0.2,0.851,0.05).tolist()
    
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2,2)

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
#         x = F.tanh(x)
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))

        return x
    
def predict(x):
        #Apply softmax to output. 
    pred = x
    #print(F.softmax(x))
    ans = []

    for t in pred:
        if(t[0].data.cpu().numpy()[0] > t[1].data.cpu().numpy()[0]):
            ans.append(0)
        else:
            ans.append(1)
                
#         print(self.forward(x).size())
#         return torch.tensor(ans).unsqueeze(1)
   # return torch.tensor(ans)
    return pred

start = timeit.default_timer()
    
prediction_y = []   

# for i in range(len(X_train_datasets_resampled)):
for i in range(1):
    print("Datasets" , i)
    for j in range(21):
        Xtrain = Variable(torch.Tensor(X_train_datasets_resampled[i*21+j].tolist()).cuda())

        ytrain = Variable(torch.Tensor(y_train_datasets_resampled[i*21+j].tolist()).cuda())
        if(len(ytrain.size()) > 1):
            ytrain = ytrain.squeeze(1)
        Xtest = Variable(torch.Tensor(X_test_datasets_resampled[i*21+j].tolist()).cuda())
        print("X train" , str(Xtrain.size()))
        print("y train" , str(torch.unsqueeze(ytrain,1).size()))
        print("X test" , str(Xtest.size()))


        net = Net(n_feature=9, n_hidden1=20 , n_hidden2=20, n_output=2)     # define the network
        net.cuda()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = torch.nn.CrossEntropyLoss()  # this is for regression mean squared loss


        # train the network
        for epoch in range(1000):

            
            optimizer.zero_grad()   # clear gradients for next train
            
            prediction = net(Xtrain.cuda())     # input x and predict based on x
            

            loss = loss_func(prediction.cuda(), ytrain.long().cuda())     # must be (1. nn output, 2. target)

            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            if epoch % 900 == 0:
                print('epoch {}'.format(epoch)) 

 


        pred_prob_y = F.softmax(predict(net(Xtest.cuda())))

        print("First: ",pred_prob_y)
        pred_prob_y = pred_prob_y[:,1]
        print("Second : ",pred_prob_y)
 

        for t in threshold:
            pred_y = pred_prob_y >=t
            print(pred_y)
            prediction_y.append(pred_y)

stop = timeit.default_timer()
print(" Time taken: " , stop - start)  
print(prediction_y)     
torch.save(prediction_y,"../Code_github/datasets_y_prediction.pt")
