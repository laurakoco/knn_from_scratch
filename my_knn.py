
import numpy as np
import os
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris

class knn_from_scratch():

    def __init__(self, number_neighbors_k, distance_parameter_p):

        self.k = number_neighbors_k
        self.p = distance_parameter_p

    # add training set data to object
    def fit(self, x, labels):

        self.x = x
        self.labels = labels
        self.unique_labels = list(set(labels)) # store unique labels

    # calculate distance between new data point and each training set data point
    # find k smallest distances
    def predict(self, new_x):

        predict = [] # prediction list
        p = self.p
        k = self.k
        training_x = self.x

        for x in new_x: # for each element in x_list

            dist = [] # distance list

            for row in training_x: # for each data point in training set
                some_dist = distance.minkowski(row, x, p) # get minkowski distance between data point and new data point
                # p = 1 manhattan "city block" distance
                # p = 2 euclidian distance
                dist.append(some_dist)

            dist_array = np.asarray(dist)
            nearest_neighbor_idx = dist_array.argsort()[:k] # get index of k smallest distances (k nearest neighbors)

            # keep code as generic as possible
            label_list = self.unique_labels
            label_dict = {} # use dictionary to keep track of count for each label
            for i in label_list:
                label_dict[i] = 0

            for i in nearest_neighbor_idx:
                 this_label = self.labels[i]
                 for key in label_dict:
                     if this_label == key:
                         label_dict[key] += 1

            # find most frequent label
            max = 0
            for key in label_dict:
                if label_dict[key] > max:
                    prediction = key
                    max = label_dict[key]

            predict.append(prediction)

        return predict

    def draw_decision_boundary(self, new_x):

        pass

        # numpy pdf - pg
        # linspace a - b
        #
        # construct mesh of points by classifier each and plot them

def filter_by_date(df, start_date, end_date):

    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]

    return df

if __name__ == "__main__":

    # import some data to play with
    iris = load_iris()

    # put into dataframe
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])

    x = iris.data[:,:2] # we only take the first two features
    y = iris.target

    # scale x
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)

    # split into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=4)

    # custom_knn(k_nearest_neighbors, Minkowski p_norm distance)
    # p = 1 manhattan "city block" distance
    # p = 2 euclidian distance
    my_knn = knn_from_scratch(5,2) # (k=5, p=2)
    my_knn.fit(x_train, y_train)
    y_pred = my_knn.predict(x_test)

    acc = metrics.accuracy_score(y_pred, y_test)
    conf_matrix = metrics.confusion_matrix(y_pred, y_test)

    print(acc)
    print(conf_matrix)

    tp = conf_matrix[0, 0]
    tn = conf_matrix[1, 1]
    fp = conf_matrix[1, 0]
    fn = conf_matrix[0, 1]

    sens = tp / (tp + fn)  # recall
    spec = tn / (tn + fp)  # specificity, tnr

    print('sensitivity (tpr): ' + str(round(sens, 2)))
    print('specificity (tnr): ' + str(round(spec, 2)))

