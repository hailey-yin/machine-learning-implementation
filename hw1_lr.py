from __future__ import division, print_function

from typing import List

import numpy
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class LinearRegression:
    
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]

    def train(self, features: List[List[float]], values: List[float]):
        ones = numpy.ones(len(features));
        X = numpy.c_[ones,features]
        self.w = numpy.dot(numpy.linalg.pinv(numpy.dot(numpy.transpose(X), X)), numpy.dot(numpy.transpose(X), values))

    def predict(self, features: List[List[float]]) -> List[float]:
        X = []
        for row in features:
            temp = row[:]
            temp.insert(0,1)
            X.append(temp)
        return numpy.dot(X, self.w)

    def get_weights(self) -> List[float]:
        return self.w


class LinearRegressionWithL2Loss:

    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]

    def train(self, features: List[List[float]], values: List[float]):
        ones = numpy.ones(len(features));
        X = numpy.c_[ones,features]
        I = numpy.identity(len(X[0]))
        self.w = numpy.linalg.pinv(X.transpose().dot(X)+self.alpha*I).dot(X.transpose()).dot(values)

    def predict(self, features: List[List[float]]) -> List[float]:
        ones = numpy.ones(len(features));
        X = numpy.c_[ones,features]
        return numpy.dot(X, self.w)

    def get_weights(self) -> List[float]:
        return self.w


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
