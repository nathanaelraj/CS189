from numpy.random import uniform
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys

from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class SVM_Model(): 

	def __init__(self,class_labels,projection=None):

		###SLACK HYPERPARAMETER
		self.C = 1.0
		



	def train_model(self,X,Y): 
            self.clf = LinearSVC(random_state=0)
            self.clf.fit(X,Y)
		
		
		

	def eval(self,x):
        	return self.clf.predict(x)




	