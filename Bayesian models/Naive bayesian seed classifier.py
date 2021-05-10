import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

seed_data = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, dtype=np.float64)
y = seed_data[7].astype('int32')
np.random.seed(100)

def random_sort_data():
    data = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None, dtype=np.float64)
    data = data.reindex(np.random.permutation(data.index))
    return data


def plots():
    plt.scatter(seed_data[0], seed_data[1], alpha=0.8, s=50, c=seed_data[7], cmap='viridis')
    plt.xlabel('Area')
    plt.ylabel('Perimeter')
    plt.show()
    plt.scatter(seed_data[3], seed_data[4], alpha=0.8, s=50, c=seed_data[7], cmap='viridis')
    plt.show()
    plt.scatter(seed_data[1], seed_data[4], alpha=0.8, s=50, c=seed_data[7], cmap='viridis')
    plt.show()
    plt.scatter(seed_data[0], seed_data[2], alpha=0.8, s=50, c=seed_data[7], cmap='viridis')
    plt.show()
    plt.scatter(seed_data[0], seed_data[2], alpha=0.8, s=50, c=seed_data[7], cmap='viridis')
    plt.show()


class NaiveBayesClassifier:

    def _data(self, features, group):
        self._samples_number, features_number = features.shape
        self._classes = np.unique(group)
        distinct_classes = len(self._classes)
        self._probability_of_class = np.zeros(distinct_classes)
        self._mean = np.zeros((distinct_classes, features_number))
        self._variance = np.zeros((distinct_classes, features_number))

        if self._classes[0] == 1:
            self._data_correction = 1
        else:
            self._data_correction = 0

    def fit(self, features, group):
        self._data(features, group)
        for distinct in self._classes:
            features_of_class = features[distinct == group]
            self._probability_of_class[distinct - self._data_correction] = features_of_class.shape[
                                                                               0] / self._samples_number
            self._mean[distinct - self._data_correction, :] = features_of_class.mean(axis=0)
            self._variance[distinct - self._data_correction, :] = features_of_class.var(axis=0)

    def predict(self, features):
        group_prediction = [self._predict(x) for x in features]
        return np.array(group_prediction)

    def _predict(self, x):
        class_probability_table = []
        for idx in self._classes:
            prior = np.log(self._probability_of_class[idx - self._data_correction])
            class_conditional = np.sum(np.log(self._pdf(idx - self._data_correction, x)))
            new_class_probability = prior + class_conditional
            class_probability_table.append(new_class_probability)

        return self._classes[np.argmax(class_probability_table)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._variance[class_idx]
        return np.exp(-(x - mean) ** 2 / 2 * var) / np.sqrt(2 * np.pi * var)


def stats(y_test, predictions):
    cnf_matrix = metrics.confusion_matrix(y_test, predictions)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Accuracy = (TP + TN) / (TP + FP + FN + TN)

    print(Recall)
    print(Precision)
    print(Accuracy)


seed_data.drop([7], axis=1, inplace=True)
seed_data = seed_data.to_numpy()

min = 10
max = 92
# max = 10
for i in range(min, max, 10):
    print(i * 0.01)
    X_train, X_test, y_train, y_test = train_test_split(seed_data, y, test_size=i * 0.01)
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    stats(y_test, predictions)
    print()

seed_data = random_sort_data()
y = seed_data[7].astype('int32')
seed_data.drop([7], axis=1, inplace=True)
seed_data = seed_data.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(seed_data, y, test_size=0.3)

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
stats(y_test, predictions)
