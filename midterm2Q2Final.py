import numpy as np
import matplotlib as plt
from numpy.linalg import inv
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

#Loading the training data
train_data = np.loadtxt('Q2train.csv',delimiter=',')
test_data = np.loadtxt('Q2test.csv',delimiter=',')

z= train_data

#Plotting the required dataset
dt = plt.pyplot.plot(z[:,1],z[:,2],'.',ms=10,linestyle='--')
# labeling the x axis 
plt.pyplot.xlabel('Latitude',fontsize=15) 
# labelingthe y axis 
plt.pyplot.ylabel('Longitude',fontsize=15)   
# Title 
plt.pyplot.title('Plot of Training DataSet')
plt.pyplot.show()

def train_Kalman(train_data,test_data,Kk,Ss):
    t=2#defining time
    
    #For Finding the estimated sequence
    
    #Initial State Matrix
    
    X = np.array([0,0,0,0,0,0])
    A=np.array([[1,t,0.5*t**2,0,0,0],
               [0,1,t,0,0,0],
               [0,0,1,0,0,0],
               [0,0,0,1,t,0.5*t**2],
               [0,0,0,0,1,t],
               [0,0,0,0,0,1]])
    
    C=np.array([[1,0,0,0,0,0],
                [0,0,0,1,0,0]
                ])
    pI = [[0.001,0,0,0,0,0],
          [0,0.001,0,0,0,0],
          [0,0,0,0,0,0],
          [0,0,0,0.001,0,0],
          [0,0,0,0,0.001,0],
          [0,0,0,0,0,0]]
    
    K_noise=Kk;
    S_noise=Ss;
    
    Q = K_noise*np.eye(6)
    R = S_noise*np.eye(2)
    
    X_pred=[]
    X_est=[]
    
    for i in range(100):
        X_pred = np.dot(A,X)#xt+1 = A*xt
        P_upd = np.dot(np.dot(A,pI),A.T) + Q#Pt+1=A*Pt*A' + Q
        K = np.dot(np.dot(P_upd,C.T),inv(np.dot(np.dot(C,P_upd),C.T)+R))#Kt+1 = Pt+1*C'(C*Pt+1C' + R)
        data_true=[train_data[i,1],train_data[i,0]*2,2,train_data[i,2],train_data[i,0]*2,2]
        Y = np.dot(C,data_true)
        Xp = X_pred + np.dot(K,(Y-np.dot(C,X_pred)))
        X_est.append(Xp)
        P = P_upd - np.dot(K,np.dot(C,P_upd))#updating pt+1
        X = Xp
        pI = P
        
    return X_est

def test_Kalman(train_data,test_data,Kk,Ss):
    t=2#defining time
    
    #For Finding the estimated sequence
    
    #Initial State Matrix
    
    A=np.array([[1,t,0.5*t**2,0,0,0],
               [0,1,t,0,0,0],
               [0,0,1,0,0,0],
               [0,0,0,1,t,0.5*t**2],
               [0,0,0,0,1,t],
               [0,0,0,0,0,1]])
    
    C=np.array([[1,0,0,0,0,0],
                [0,0,0,1,0,0]
                ])
    pI = [[0.1,0,0,0,0,0],
          [0,0.1,0,0,0,0],
          [0,0,0,0,0,0],
          [0,0,0,0.1,0,0],
          [0,0,0,0,0.1,0],
          [0,0,0,0,0,0]]
    
    K_noise=Kk;
    S_noise=Ss;
    
    Q = K_noise*np.eye(6)
    R = S_noise*np.eye(2)
    X = np.array([0,0,0,0,0,0])
    X_pred=[]
    X_est=[]
    
    for i in range(100):
        X_pred = np.dot(A,X)
        P_upd = np.dot(np.dot(A,pI),A.T) + Q
        K = np.dot(np.dot(P_upd,C.T),inv(np.dot(np.dot(C,P_upd),C.T)+R))
        data_true=[train_data[i,1],train_data[i,0]*2,2,train_data[i,2],train_data[i,0]*2,2]
        Y = np.dot(C,data_true)
        Xp = X_pred + np.dot(K,(Y-np.dot(C,X_pred)))
        X_est.append(Xp)
        P = P_upd - np.dot(K,np.dot(C,P_upd))
        X = Xp
        pI = P
    
    X_est_Test = []
    X2=[0,0,0,0,0,0]
    for i in range(100):
        X_pred = np.dot(A,X2)
        P_upd = np.dot(np.dot(A,pI),A.T) + Q
        K = np.dot(np.dot(P_upd,C.T),inv(np.dot(np.dot(C,P_upd),C.T)+R))
        data_true=[test_data[i,1],test_data[i,0]*2,2,test_data[i,2],test_data[i,0]*2,2]
        Y = np.dot(C,data_true)
        Xp = X_pred + np.dot(K,(Y-np.dot(C,X_pred)))
        X_est_Test.append(Xp)
    
    return X_est_Test

def crossVal(train_data,test_data,Kk,Ss):
    X_est_Test = np.array(test_Kalman(train_data,test_data,Kk,Ss))
    crossVal = 0
    for i in range (100):
        cv_mea = [test_data[i,1],2,2,test_data[i,2],2,2]
        crossVal = crossVal + (abs(X_est_Test[i][0]-cv_mea[0]) + abs(X_est_Test[i][3]-cv_mea[3]))**2
    crossVal_Score = crossVal/100
    
    return crossVal_Score

scrs=[]

#Getting the estimated sequence from Trianing data
xe_temp1 = train_Kalman(train_data,test_data,9,1)
xe_train=np.zeros([100,2])
C1=np.array([[1,0,0,0,0,0],
             [0,0,0,1,0,0]])
for i in range(100):
    xe_train[i,:]=np.squeeze(np.dot(C1,np.expand_dims(xe_temp1[i],1)))#Estimated [h,b] for training data
#Getting Estimated Sequence from Test Data
xe_temp2 = test_Kalman(train_data,test_data,9,1)
xe_test=np.zeros([100,2])
C1=np.array([[1,0,0,0,0,0],
             [0,0,0,1,0,0]])
for i in range(100):
    xe_test[i,:]=np.squeeze(np.dot(C1,np.expand_dims(xe_temp2[i],1)))

#Generating CrossValidation
for k in range(1,20):
    for s in range(1,20):        
        scr=crossVal(train_data,test_data,k,s)
        scrs.append([k,s,scr])
#Finding the Minimum K,S from the Generated Scores

scrs = np.array(scrs)
scrs_min = np.argmin(scrs,axis =0)[2]
print("Optimal (K,S,Score) is : ", scrs[scrs_min,:])

#Plotting the contour 
list_1 = np.linspace(1,20,19)
list_2 = np.linspace(1,20,19)

X,Y = np.meshgrid(list_1,list_2)
Z = scrs[:,2].reshape(19,19)
plt.pyplot.contour(X,Y,Z)
plt.pyplot.plot(19,1,'o',ms=10)
plt.pyplot.xlabel('Value of K',fontsize=15)
plt.pyplot.ylabel('Value of S',fontsize=15)
plt.pyplot.title('Plot of (K,S) with Respective Score',fontsize=18)
plt.pyplot.show()

#Plotting the train,test,kalman_train,kalman_test
plt.pyplot.plot(train_data[:,1],train_data[:,2],'.',ms=10)
plt.pyplot.plot(test_data[:,1],test_data[:,2],'.',ms=10)
plt.pyplot.plot(xe_train[:,0],xe_train[:,1],linestyle='--')
plt.pyplot.plot(xe_test[:,0],xe_test[:,1],linestyle='--')
plt.pyplot.xlabel('Latitude',fontsize=15)
plt.pyplot.ylabel('Longitutde',fontsize=1)
plt.pyplot.title('Plot of Training and Test Samples with Estimated Train and Test',fontsize=18)
plt.pyplot.legend(['Train','Test','Estimated Train','Estimated Test'])
plt.pyplot.show()