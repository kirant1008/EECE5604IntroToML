# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:06:13 2019

@author: kiran
"""
import numpy as np
import matplotlib as plt



#Specifying Standard Deviation
stand_dev=1

#Number of samples
no_Sam=10

#Defining W true = [ a b c d]
a= np.random.rand(1,3)
a=np.array(a)
W_True = np.append(a,1)
W_True = np.array([W_True])

#Specifying Range of Gamma
no_Samples = 10000
no_gamm = 100
gamm = np.array([pow(10,(np.linspace(-2,2,no_gamm)))])

#To store Squared Error Values
sqrd_error = np.zeros([no_Samples,no_gamm])

for i in range(0,no_gamm):
    gamma = gamm[0][i]
    for j in range(0,no_Samples):
        
        #Generate Data
        x= -1 + 2*np.random.rand(1,no_Sam)
        #Generating Noise
        v=stand_dev*np.random.rand(1,no_Sam)
        
        #Generating X^3 X^2 X and 1
        
        A= np.ones([1,no_Sam])
        B= np.zeros([3,no_Sam])
        X = np.concatenate([A,B])
        
        # X = [ 1 x x^2 x^3]
        
        X[1,:] = pow(x,1)
        X[2,:] = pow(x,2)
        X[3,:] = pow(x,3)
        
        #Calculating y=w'*x+c
        
        y= np.matmul(W_True,X) + v
        

        
        #Estimate parameters with MAP using Y with Noise
        c=pow((stand_dev/gamma),2)
        n=c*np.eye(4,4)
        z= np.matmul(X,np.transpose(X)) + n
        z= np.linalg.inv(z)
        
        p= np.matmul(X,np.transpose(y))
        w_map = np.matmul(z,p)
        error = np.linalg.norm(W_True-w_map)
        sqrd_error[j,i] = pow(error,2)
print(sqrd_error.shape)

#Using percentile calcualting minimum 25% median 75% and Maximum
op_errors = np.percentile(sqrd_error,[0,25, 50, 75, 100],1)

#Ploting ths squared error values 
prcnt_plt=plt.pyplot.loglog(gamm[0,:],op_errors[0,:],label='Min')
prcnt_plt=plt.pyplot.loglog(gamm[0,:],op_errors[1,:],label='25%')
prcnt_plt=plt.pyplot.loglog(gamm[0,:],op_errors[2,:],label='Med')
prcnt_plt=plt.pyplot.loglog(gamm[0,:],op_errors[3,:],label='75%')
prcnt_plt=plt.pyplot.loglog(gamm[0,:],op_errors[4,:],label='Max')

# labeling the x axis 
plt.pyplot.xlabel('Gamma') 
# labelingthe y axis 
plt.pyplot.ylabel('Squared Error')   
#Giving Legend
plt.pyplot.legend( )
# Title 
plt.pyplot.title('Plot of Minimum 25% 75% and Maximum of the Squared Values')

