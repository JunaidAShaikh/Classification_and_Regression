import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    
    N=X.shape[0]
    d = X.shape[1]
    out = y.reshape(y.size)
    
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    AllClasses = np.unique(y)
    k = len(AllClasses)
    means = np.zeros((d,k))
    covmat = []       

    for i in range (k):
        rows = X[out == AllClasses[i]]   ## get all rows of this class
        means[:, i] = np.mean(rows, 0)
            
    covmat = (np.cov(X.T)) 

    return means,covmat
    

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    N=X.shape[0]
    d = X.shape[1]
    out = y.reshape(y.size)
    
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    AllClasses = np.unique(y)
    k = len(AllClasses)
    means = np.zeros((d,k))
    covmat = []       

    for i in range (k):
        rows = X[out == AllClasses[i]]   ## get all rows with value of y AllClasses[i]
        means[:, i] = np.mean(rows, 0)
        covmat.append(np.cov(rows.T)) 

    return means,covmat

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #return acc,ypred
    
    determinant = np.linalg.det(covmat)
    inversecovariance = np.linalg.inv(covmat)

    prediction = np.empty((Xtest.shape[0], means.shape[1]))
    
    for i in range(0,means.shape[1]):
        xminusmean = (Xtest - means[:, i])
        #print(xminusmean)
        #print(means[:, i].shape)
        #print(means.shape)
        dot = np.dot(inversecovariance, xminusmean.T)   #2x2 , 2x100
        powerofexp = np.sum(xminusmean * dot.T, 1)  ## sum eover rows
        pred = np.exp(-(1/2) * powerofexp)
        prediction[:, i] = pred *(1/np.power(determinant, 2))

    ypred= np.argmax(prediction, 1)  #get best label
    ypred = ypred + 1
    ytest = ytest.reshape(ytest.size)

    matchingResults=0
    for i in range(0,ytest.size):
        if ytest[i] == ypred[i]:
            matchingResults=matchingResults+1
    
    acc = 100 * (matchingResults/Xtest.shape[0])
  
    return acc,ypred
    
def qdaTest(means,covmats,Xtest,ytest):
    #print(means)
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #return acc,ypred

    prediction = np.empty((Xtest.shape[0], means.shape[1]))
    
    for i in range(0,means.shape[1]):
        xminusmean = (Xtest - means[:, i])
        determinant = np.linalg.det(covmats[i])
        inversecovariance = np.linalg.inv(covmats[i])

        dot = np.dot(inversecovariance, xminusmean.T)   #2x2 , 2x100
        powerofexp = np.sum(xminusmean * dot.T, 1)  ## sum eover rows
        pred = np.exp(-(1/2) * powerofexp)
        prediction[:, i] = pred *(1/np.power(determinant, 2))

    ypred= np.argmax(prediction, 1)  #get best label
    ypred = ypred + 1
    ytest = ytest.reshape(ytest.size)

    matchingResults=0
    for i in range(0,ytest.size-1):
        if ytest[i] == ypred[i]:
            matchingResults=matchingResults+1
    
    acc = 100 * (matchingResults/ytest.size)
    return acc,ypred


def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    # IMPLEMENT THIS METHOD  
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), (np.transpose(X))), y)
    #print("w = ",np.shape(w))
    return w
def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    w = np.dot(np.dot(np.linalg.inv((lambd*np.identity(np.shape(X)[1])) + (np.dot(X.T, X))), X.T), y)                                        
    return w

def testOLERegression(w,Xtest,ytest):
  # Inputs:
   # w = d x 1
   # Xtest = N x d
   # ytest = X x 1
   # Output:
   # mse
   
   # IMPLEMENT THIS METHOD
    #print("test = ",np.shape(Xtest), "no of rows = ",np.shape(Xtest)[0])
    
    mse = (1/np.shape(Xtest)[0])*np.dot(np.transpose(ytest - np.dot(Xtest, w)), (ytest - np.dot(Xtest,w)))
    #print(mse)
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    
    #Xis 242*65 , w vector 
    w = w.reshape(np.shape(w)[0],1)
    xw=np.dot(X,w)
    yminusxw=y-xw
    
    error = (0.5 * np.dot(yminusxw.T,yminusxw)) + (0.5 * lambd * np.dot(w.T,w))
    error_grad = (-1*np.dot(X.T, y-np.dot(X,w))) +  lambd * w
    error = error.flatten()
    error_grad = error_grad.flatten()
    #print("error =" , np.shape(error))
    #print("error_grad = ", np.shape(error_grad))                                         
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    

    Xd = np.ones((np.shape(x)[0], p + 1))
    for i in range(p):
        Xd[:, i+1] = np.power(x,i+1)
    return Xd


# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)


trainingMLE= testOLERegression(w,X,y)
trainingMLE_i= testOLERegression(w_i,X_i,y)
print('MSE without intercept for testing'+str(mle))
print('MSE with intercept for testing'+str(mle_i))
print('MSE without intercept for training'+str(trainingMLE))
print('MSE with intercept for training'+str(trainingMLE_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 40}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
#lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
lambda_opt = lambdas[np.argmin(mses3)]
#print(lambda_opt)
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
