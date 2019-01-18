
from numpy.random import uniform
import random
import time


import numpy as np
import numpy.linalg as LA

import sys

from sklearn.linear_model import Ridge

from utils import create_one_hot_label


class Ridge_Model(): 

	def __init__(self,class_labels):

		###RIDGE HYPERPARAMETER
		self.lmda = 1.0
		

	def train_model(self,X,Y): 
            X = np.array(X)
            Y = create_one_hot_label(Y,3)
            self.clf = Ridge(alpha=self.lmda)
            self.clf.fit(X,Y)

	def eval(self,x):
            result = self.clf.predict(x)
            return np.argmax(result)
            
                      




	