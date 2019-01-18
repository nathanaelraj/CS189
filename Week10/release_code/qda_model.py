import random
import time


import numpy as np
import numpy.linalg as LA

import math
from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections

from utils import subtract_mean_from_data
from utils import compute_covariance_matrix

class QDA_Model(): 

	def __init__(self,class_labels):

		###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
		self.reg_cov = 0.01
		self.NUM_CLASSES = len(class_labels)



	def train_model(self,X,Y): 
            """
            self.muj = []
            self.cov_XX = []
            X0 = X[:324]
            mu0 = np.mean(X0,axis=0)
            X1 = X[325:633]
            mu1 = np.mean(X1,axis=0)
            X2 = X[634:]
            mu2 = np.mean(X2,axis=0)
            self.muj.append(mu0)
            self.muj.append(mu1)
            self.muj.append(mu2)
            X_bar_0 = subtract_mean_from_data(X0, Y)
            X_bar_1 = subtract_mean_from_data(X1, Y)
            X_bar_2 = subtract_mean_from_data(X2, Y)
            self.cov_XX.append(compute_covariance_matrix(X_bar_0[0],X_bar_0[0])+np.identity(len(np.array(X0).T))*self.reg_cov)
            self.cov_XX.append(compute_covariance_matrix(X_bar_1[0],X_bar_1[0])+np.identity(len(np.array(X0).T))*self.reg_cov)
            self.cov_XX.append(compute_covariance_matrix(X_bar_2[0],X_bar_2[0])+np.identity(len(np.array(X0).T))*self.reg_cov)
            """
            self.muj = []
            self.cov_XX = []
            j = 0
            k = 0
            val = Y[0]
            for i in range(self.NUM_CLASSES):
                while(j<len(Y)-1 and val==Y[j]):
                    j = j + 1
                X0 = X[k:j-1]
                mu0 = np.mean(X0,axis=0)
                self.muj.append(mu0)
                X_bar_0 = subtract_mean_from_data(X0,Y)
                cov_XX_0 = compute_covariance_matrix(X_bar_0[0],X_bar_0[0]) + np.identity(len(np.array(X0).T))*self.reg_cov
                self.cov_XX.append(cov_XX_0)
                k = j
                val = Y[j]

	def eval(self,x):
            results = []
            for i,j in zip(self.muj,self.cov_XX):
                if(det(j)>0):
                    result = -(x-i).T.dot(inv(j)).dot(x-i) - math.log(det(j))
                else:
                    result = -(x-i).T.dot(inv(j)).dot(x-i)
                results.append(result)
            return np.argmax(results)


	