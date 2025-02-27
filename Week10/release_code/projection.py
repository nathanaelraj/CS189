
from numpy.random import uniform
from numpy.random import randn
import random
import time

import matplotlib.pyplot as plt

from scipy.linalg import eig
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import svd

from utils import create_one_hot_label
from utils import subtract_mean_from_data
from utils import compute_covariance_matrix

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import svd



class Project2D(): 

	'''
	Class to draw projection on 2D scatter space
	'''

	def __init__(self,projection, clss_labels):

		self.proj = projection
		self.clss_labels = clss_labels


	def project_data(self,X,Y,white=None): 

		'''
		Takes list of state space and class labels
		State space should be 2D
		Labels shoud be int
		'''

		p_a = []
		p_b = []
		p_c = []

		# Project all Data
		proj = np.matmul(self.proj,white)
	
		X_P = np.matmul(proj,np.array(X).T)
		
		for i in range(len(Y)):

			if Y[i] == 0: 
				p_a.append(X_P[:,i])
			elif Y[i] == 1: 
				p_b.append(X_P[:,i])
			else:
				p_c.append(X_P[:,i])

		
		p_a = np.array(p_a)
		p_b = np.array(p_b)
		p_c = np.array(p_c)

		plt.scatter(p_a[:,0],p_a[:,1],label = 'apple')
		plt.scatter(p_b[:,0],p_b[:,1],label = 'banana')
		plt.scatter(p_c[:,0],p_c[:,1],label = 'eggplant')

		plt.legend()

		plt.show()



class Projections():

	def __init__(self,dim_x,classes):

		'''
		dim_x: the dimension of the state space x 
		classes: The list of class labels 
		'''

		self.d_x = dim_x
		self.NUM_CLASSES = len(classes)
		


	def get_random_proj(self):
            A = np.random.normal(0,1,(2,729))
            return A



	def pca_projection(self,X,Y):
            #mux = np.mean(X,axis=0)
            X_bar = subtract_mean_from_data(X,Y)
            cov_XX = compute_covariance_matrix(X_bar[0],X_bar[0])
            U,s,V = svd(cov_XX)
            return U.T[:2]



	def cca_projection(self,X,Y,k=2):
            reg = 1e-5
            Y = np.array(Y)
            one_hot_y = create_one_hot_label(Y,self.NUM_CLASSES)
            X_bar = subtract_mean_from_data(X,Y)
            Y_bar = subtract_mean_from_data(one_hot_y,X)
            cov_XX = compute_covariance_matrix(X_bar[0],X_bar[0])
            cov_XX += np.identity(729) * (reg)
            cov_YY = compute_covariance_matrix(Y_bar[0],Y_bar[0])
            cov_XY = compute_covariance_matrix(X_bar[0],Y_bar[0])
            cov_XX_inv = inv(sqrtm(cov_XX))
            cov_YY_inv = inv(sqrtm(cov_YY))
            ccm = cov_XX_inv.dot(cov_XY).dot(cov_YY_inv)
            U,s,V = svd(ccm)
            return U.T[:k],cov_XX_inv


	def project(self,proj,white,X):
		'''
		proj, numpy matrix to perform projection
		whit, numpy matrix to perform whitenting
		X, list of states
		'''

		proj = np.matmul(proj,white)
	
		X_P = np.matmul(proj,np.array(X).T)

		return list(X_P.T)



if __name__ == "__main__":

	X = list(np.load('little_x_train.npy'))
	Y = list(np.load('little_y_train.npy'))

	CLASS_LABELS = ['apple','banana','eggplant']

	feat_dim = max(X[0].shape)
	projections = Projections(feat_dim,CLASS_LABELS)


	rand_proj = projections.get_random_proj()
	# Show Random 2D Projection
	proj2D_viz = Project2D(rand_proj,CLASS_LABELS)
	proj2D_viz.project_data(X,Y, white = np.eye(feat_dim))

	#PCA Projection 
	pca_proj = projections.pca_projection(X,Y)
	
	#Show PCA 2D Projection
	proj2D_viz = Project2D(pca_proj,CLASS_LABELS)
	proj2D_viz.project_data(X,Y, white = np.eye(feat_dim))

	#CCA Projection 
	cca_proj,white_cov = projections.cca_projection(X,Y)
	#Show CCA 2D Projection
	proj2D_viz = Project2D(cca_proj,CLASS_LABELS)
	proj2D_viz.project_data(X,Y,white = white_cov)
