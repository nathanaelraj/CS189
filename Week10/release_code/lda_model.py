import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections

from utils import create_one_hot_label
from utils import subtract_mean_from_data
from utils import compute_covariance_matrix


class LDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.001
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
            """
            X_bar = subtract_mean_from_data(X,Y)
            cov_XX = compute_covariance_matrix(X_bar[0],X_bar[0])
            cov_XX += np.identity(len(np.array(X_bar).T)) * self.reg_cov
            self.cov_XX = cov_XX
            self.muj = []
            X0 = X[:324]
            mu0 = np.mean(X0,axis=0)
            print(mu0)
            X1 = X[325:633]
            mu1 = np.mean(X1,axis=0)
            X2 = X[634:]
            mu2 = np.mean(X2,axis=0)
            self.muj.append(mu0)
            self.muj.append(mu1)
            self.muj.append(mu2)
            
            """
            X_bar = subtract_mean_from_data(X,Y)
            cov_XX = compute_covariance_matrix(X_bar[0],X_bar[0])
            cov_XX += np.identity(len(np.array(X).T)) * self.reg_cov
            self.cov_XX = cov_XX
            self.muj = []
            j=0
            k=0
            val = Y[0]
            for i in range(self.NUM_CLASSES):
                print(i)
                while(j<len(Y)-1 and val==Y[j]):
                        j = j+1
                X0 = X[k:j-1]
                mu0 = np.mean(X0,axis=0)
                self.muj.append(mu0)
                k = j
                val = Y[j]
                              

           

	def eval(self,x):
            results = []
            for i in self.muj:
                result = -(x-i).T.dot(inv(self.cov_XX)).dot(x-i)
                results.append(result)
            return np.argmax(results)
            


	