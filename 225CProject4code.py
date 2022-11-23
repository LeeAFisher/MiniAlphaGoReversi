import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
#matplotlib inline

# Read the data
train_data = pd.read_csv('gamedata.csv')
#test_data = pd.read_csv("test.csv")


# Set up the data
y_train = train_data.columns.values.astype(float).astype(int)
X_train = train_data.values
X_train = X_train.T


# relu activation function
# THE fastest vectorized implementation for ReLU
def relu(x):
    x[x<0]=0
    return x

def h(X,W,b):
    '''
    This is a forward calculation which handles any amount of hidden layers
    '''
    total = 1
    a = [1]*(len(W)+1)
    z = [1]*(len(W))
    # layer 1 = input layer
    a[0]= X
    
    for i in range(len(W)-1):
        z[i] = np.matmul(a[i],W[i])+ b[i]
        a[i+1] = relu(z[i])
    
    z[-1] = np.matmul(a[-2],W[-1])
    #This is the softmax function on the activation layer
    s = np.exp(z[-1])
    total = np.maximum(np.sum(s, axis=1).reshape(-1,1), sys.float_info.min)
    a[-1] = s/total
    return a[-1]

def loss(y_pred,y_true):
    '''
    Loss function: cross entropy with an L^2 regularization
    y_true: ground truth, of shape (N, )
    y_pred: prediction made by the model, of shape (N, K) 
    N: number of samples in the batch
    K: global variable, number of classes
    '''
    global K 
    K = 64
    N = len(y_true)
    # loss_sample stores the cross entropy for each sample in X
    # convert y_true from labels to one-hot-vector encoding
    y_true_one_hot_vec = (y_true[:,np.newaxis] == np.arange(K))
    loss_sample = (y_pred * y_true_one_hot_vec).sum(axis=1)
    # loss_sample is a dimension (N,) array
    # for the final loss, we need take the average
    return -np.mean(loss_sample)

#This function does backpropagation and it accepts a general layer structure
def backprop(W,b,X,y,alpha):
    '''
    Step 1: explicit forward pass h(X;W,b)
    Step 2: backpropagation for dW and db
    '''
    K = 64
    N = X.shape[0]
    
    a = [1]*(len(W)+1)
    z = [1]*(len(W))
    # layer 1 = input layer
    a[0]= X
    
    #This is the forward pass
    for i in range(len(W)-1):
        z[i] = np.matmul(a[i],W[i])+ b[i]
        a[i+1] = relu(z[i])
    
    z[-1] = np.matmul(a[-2],W[-1])
    #This is the softmax function on the activation layer
    s = np.exp(z[-1])
    total = np.maximum(np.sum(s, axis=1).reshape(-1,1), sys.float_info.min)
    a[-1] = s/total
    
    ### Step 2:
    
    delta = [1]*(len(W))
    grad_W = [1]*(len(W))
    
    y_one_hot_vec = (y[:,np.newaxis] == np.arange(K))
    delta[-1]= (a[-1] - y_one_hot_vec)
    grad_W[-1] = np.matmul(a[-2].T, delta[-1])
    
    #This is the backpropagation loop
    for i in range(len(W)-2, -1, -1):
        delta[i] = np.matmul(delta[i+1], W[i+1].T)*(z[i]>0)
        grad_W[i] = np.matmul(a[i].T, delta[i])

    dW = [1]*(len(W))
    for i in range(len(W)):
        dW[i] = grad_W[i]/N + alpha*W[i]
        
    db = [1]*(len(b))
    for i in range(len(b)):
        db[i] = np.mean(delta[i], axis=0)
    
    return dW, db


#This function trains the algorithm using a stochastic gradient descent
def SGD_Train(eta, alpha, gamma, eps, num_iter, X, Y, Layers, Batch):
    N = X.shape[0] #Sample size
    # initialization of random weights and bias
    np.random.seed(1127)
    W = [1]*(len(Layers)-1)
    b = [1]*(len(Layers)-2)

    for i in range(len(W)):
        W[i] = 1e-1*np.random.randn(Layers[i], Layers[i+1])

    for i in range(len(W)-1):
        b[i] = np.random.randn(Layers[i+1])


    gW= np.ones(len(W)) 
    gb =np.ones(len(b)) 
    etaW = np.ones(len(W))
    etab= np.ones(len(b))

    for i in range(num_iter):
        #Uncomment for SGD
        random_indices = np.random.choice(np.arange(N), Batch, False)
        X_sample = X[random_indices]
        y_sample = Y[random_indices]
        dW, db = backprop(W,b,X_sample,y_sample,alpha)
    
        #Uncomment for FGD
        """
        dW, db = backprop(W,b,X_train,y_train,alpha)
        """
        for j in range(len(W)):
                gW[j] = gamma*gW[j] + (1-gamma)*np.sum(dW[j]**2)
                etaW[j] = eta/np.sqrt(gW[j] + eps)
                W[j] -= etaW[j]*dW[j]
                
        for j in range(len(b)):
                gb[j] = gamma*gb[j] + (1-gamma)*np.sum(db[j]**2)
                etab[j] = eta/np.sqrt(gb[j] + eps)
                b[j] -= etab[j]*db[j]

    return W, b

#This code is set up to run a K-fold cross validation to estimate the 
#accuracy of the training model. It can be very slow. 
def CrossValidate(K, X, Y, eta, alpha, gamma, eps, num_iter, Layers, Batch):
    Errors = np.zeros(K)
    N = X.shape[0]
    for i in range(K):
        X_fold = X[int(i/K*N):int((i+1)/K*N)]
        Y_fold = Y[int(i/K*N):int((i+1)/K*N)]
        
        X_fold_train =np.concatenate((X[:int(i/K*N)],X[int((i+1)/K*N):]))
        Y_fold_train =np.concatenate((Y[:int(i/K*N)],Y[int((i+1)/K*N):]))
        
        Fold_W, Fold_b = SGD_Train(eta, alpha, gamma, eps, num_iter, 
                                   X_fold_train, Y_fold_train, Layers, Batch)
        
        Y_pred = h(X_fold, Fold_W, Fold_b)
        
        Errors[i] = np.mean(np.argmax(Y_pred, axis=1)== Y_fold)
        print("Fold number", i+1,"has error of {:.4%}".format(Errors[i]))
    return np.mean(Errors)

eta = 5e-1
alpha = 1e-5 # regularization
gamma = 0.99 # RMSprop
eps = 1e-3 # RMSprop
num_iter = 2000 # number of iterations of gradient descent
Batch = 300 #this is the size of the batch to use in SGD
n = X_train.shape[1] # number of pixels in an image
Layers = [n, 256,256, 64] #This is the layer structure of our system
F = 3 # Number of folds in the cross validation

#The main method calls in the program
#E = CrossValidate(F, X_train, y_train, eta, alpha, gamma, eps, 
#                  num_iter, Layers, Batch)

#Here we are
W, b = SGD_Train(eta, alpha, gamma, eps, 
                 num_iter, X_train, y_train, Layers, Batch)

y_pred_final = h(X_train, W, b)

#print("Final cross-entropy loss is {:.8}".format(loss(y_pred_final,y_train)))
print("Final training accuracy is {:.4%}".format(np.mean(np.argmax(y_pred_final, axis=1)== y_train)))
#print("Final", F, "fold cross validation accuracy is {:.4%}".format(E))


# predictions
#y_pred_test = np.argmax(h(X_test,W,b), axis=1)


pd.DataFrame(W[0]).to_csv("weights1.csv",index=False)
pd.DataFrame(W[1]).to_csv("weights2.csv",index=False)
pd.DataFrame(W[2]).to_csv("weights3.csv",index=False)
pd.DataFrame(b[0]).to_csv("biases1.csv",index=False)
pd.DataFrame(b[1]).to_csv("biases2.csv",index=False)
