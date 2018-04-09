from __future__ import division, print_function

from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:    
        for iteration in range(self.max_iteration):
            sum_error = 0.0
            for i in range(len(features)):
                predict = np.dot(np.transpose(self.w), features[i])
                if int(np.sign(predict))!=labels[i] or abs(predict/np.linalg.norm(self.w,2))<abs(self.margin/2):
                    sum_error += 1
                    norm = np.linalg.norm(features[i],2)
                    self.w = np.add(self.w, labels[i]/norm*np.asarray(features[i])) 
            if sum_error==0:
                return True
        return False    
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        result = []
        for row in features:
             result.append(int(np.sign(np.dot(np.transpose(row), self.w))))                
        return result

    def get_weights(self) -> Tuple[List[float], float]:
        return self.w
    