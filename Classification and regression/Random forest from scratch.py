import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv('dataset.csv', dtype=np.float64).to_numpy()


def potential_splits(dataset):
    number_of_rows, number_of_columns = dataset.shape
    splits_by_index = {}
    for column_index in range (number_of_columns-1):
        splits_by_index[column_index] = []
        values = dataset[:, column_index]
        unique_values = np.unique(values)
        for index in range (len(unique_values) - 1):
            mean_of_two_elements = (unique_values[index] + unique_values[index+1]) / 2
            splits_by_index[column_index].append(mean_of_two_elements)
    return splits_by_index

def calculate_entropy(splits_by_index):

    for i in splits_by_index:
        unique_data = splits_by_index[i]
        for x in range (len(unique_data)):
            print(unique_data[x])

calculate_entropy(potential_splits(data))