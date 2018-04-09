from __future__ import division, print_function

from typing import List, Callable

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:
    
    train_features = []
    train_labels = []

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.train_features = features
        self.train_labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        import operator
        result = []
        for i in range(len(features)):
            distance = []
            for j in range(len(self.train_features)):
                dis = self.distance_function(features[i], self.train_features[j])
                distance.append((self.train_labels[j], dis))
            distance.sort(key=operator.itemgetter(1))
            vote = 0
            for i in range(self.k):
                vote += distance[i][0]
            if vote<=abs(self.k/2):
                result.append(0)
            else:
                result.append(1)
        return result


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
