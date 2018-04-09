from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    squaredError = []
    for i in range(len(y_true)):
        val = y_true[i] - y_pred[i]
        squaredError.append(val * val)     
    return sum(squaredError) / len(squaredError)

def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    true_positive = 0
    false_positive = 0
    real_positive = 0
    for i in range(len(real_labels)):
        if real_labels[i]==1:
            real_positive += 1
        if predicted_labels[i]==1:
            if real_labels[i]==1:
                true_positive += 1
            else:
                false_positive += 1    
    if true_positive == 0:
        p = 0
        r = 0
    else:
        p = true_positive/(true_positive+false_positive)
        r = true_positive/real_positive
    if p==0 and r==0:
        return 0
    else:
        return 2*p*r/(p+r)


def polynomial_features(features: List[List[float]], k: int) -> List[List[float]]:
    X = []
    for row in features:
        newfeature = []
        for i in range(k):
            for item in row:
                power = np.power(item, i+1)
                newfeature.append(power)
        X.append(newfeature)
    return X 


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1)==len(point2)
    diff_square = []
    for i in range(len(point1)):
        temp = point1[i]-point2[i]
        diff_square.append(np.power(temp, 2))
    return np.sqrt(sum(diff_square))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1)==len(point2)
    inner_product = []
    for i in range(len(point1)):
        inner_product.append(point1[i]*point2[i])
    return sum(inner_product)


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    assert len(point1)==len(point2)
    diff_square = []
    for i in range(len(point1)):
        temp = point1[i]-point2[i]
        diff_square.append(np.power(temp, 2))
    return -np.exp((-1/2)*sum(diff_square))


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        normalization=[]
        for row in features:
            inner_product = inner_product_distance(row, row)
            if inner_product==0:
                normalization.append(np.zeros(len(row)))
            else:
                normalization.append(row/(np.sqrt(inner_product)))
        return normalization


class MinMaxScaler:
    min_value = [float('inf')]
    max_value = [float('-inf')]
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        if self.min_value[0]==float('inf'):
            self.min_value = np.array(features).min(axis=0)
        if self.max_value[0]==float('-inf'):
            self.max_value = np.array(features).max(axis=0)
        scaler = []
        for row in features:
            temp = []
            for i in range(len(row)):
                if self.max_value[i]-self.min_value[i]==0:
                    temp.append(0)
                else:
                    temp.append((row[i]-self.min_value[i])/(self.max_value[i]-self.min_value[i]))
            scaler.append(temp)
        return scaler

