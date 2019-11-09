# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize 

from PIL import Image
#Function is used to generate 5D data
def data_extract(image):
    data=np.zeros([321*481,5])
    k=0
    for i in range(0,321):
        for j in range(0,481):
            r=image[0,i,j]
            g=image[1,i,j]
            b=image[2,i,j]
            data[k,:]=np.array([i,j,r,g,b])
            k=k+1
    return data

#function is uesd to form the 3d image from data image
def data_form(image):
    image_3d=np.zeros([3,321,481], dtype=np.uint8)
    r=0;
    for i in range(0,321):
        for j in range(0,481):
            image_3d[0,i,j]=image[r,0]
            image_3d[1,i,j]=image[r,1]
            image_3d[2,i,j]=image[r,2]
            r=r+1
    return image_3d
        
#Function is used to label colour
def label_colour(image,labels):
    color_Label= np.zeros([321*481,3])
    for i in range(0,154401):
                if labels[i,] == 0:
                    color_Label[i,:]= [255,0,0]
                if labels[i,] == 1:
                    color_Label[i,:]= [0,255,0]
                if labels[i,] == 2:
                    color_Label[i,:]=[0,0,255]
                if labels[i,] == 3:
                    color_Label[i,:]=[255, 255, 0]
                if labels[i,] == 4:
                    color_Label[i,:]=[255, 165, 0]
                if labels[i,] == 5:
                    color_Label[i,:]=[128, 128, 128]    
    
    return color_Label

#Applying Kmeans Algorithm
def main(data_img,nu_clust):
    img1_kmeans = KMeans(n_clusters=nu_clust)
    img1_kmeans.fit(data_img)
    img1_gmm    = GaussianMixture(n_components=nu_clust)
    img1_gmm.fit(data_img)
    predictions_img1kmeans= img1_kmeans.predict(data_img)
    predictions_img1gmm=    img1_gmm.predict(data_img)
    
    #importing dataset with updated color labels
    image_labels_knn= label_colour(data_img1,predictions_img1kmeans)
    image_labels_gmm= label_colour(data_img1,predictions_img1gmm)
    
    #Kmeans iamge
    image3d_kmean = data_form(image_labels_knn);
    image3d_kmean=  np.transpose(image3d_kmean,(1, 2 ,0))
    image3d_kmean=  Image.fromarray(image3d_kmean)
    image3d_kmean.show()
    #Gmm image
    image3d_gmm = data_form(image_labels_gmm);
    image3d_gmm=  np.transpose(image3d_gmm,(1, 2 ,0))
    image3d_gmm=  Image.fromarray(image3d_gmm)
    image3d_gmm.show()    

#Importing Images in Python 
img_1=Image.open("D:\MastersECE\EECE5644IntroToML\Assignments\colorPlane.jpg")
img_2=Image.open("D:\MastersECE\EECE5644IntroToML\Assignments\colorBird.jpg")
#Transfroming Images into arrays
img1_array= np.asarray(img_1)
img1_array= np.transpose(img1_array,(2, 0, 1))
img2_array= np.asarray(img_2)
img2_array= np.transpose(img2_array,(2, 0, 1))

#Generating 5 Dimensional Feature Vector
data_img1= data_extract(img1_array)
data_img2= data_extract(img2_array)
#normalize
data_img1=normalize(data_img1)
data_img2=normalize(data_img2)
#Giving it to main function
main(data_img1,5)
main(data_img2,5)

