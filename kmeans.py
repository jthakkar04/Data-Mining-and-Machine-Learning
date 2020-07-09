import numpy as np
import pandas as pd
import math
import sys
import os

class KMeans(object):

    def __init__(self, option):
        self.centroid = None
        self.class_labels = {}
        if option == 4:
            self.d_type = 'manhattan'
        else:
            self.d_type = 'euclidean'

    def input_stream(self, data, option):
        df = pd.read_csv(data, delimiter=',', index_col=None).dropna()
        df = df[['latitude', 'longitude', 'reviewCount', 'checkins']]

        if option == 3:
            df['latitude'] = self.cluster_3_df('latitude', df)
            df['longitude'] = self.cluster_3_df('longitude', df)
            df['reviewCount'] = self.cluster_3_df('reviewCount', df)
            df['checkins'] = self.cluster_3_df('checkins', df)
        elif option == 2:
            df['reviewCount'] = np.log(df['reviewCount'])

        df = df.values

        return df

    def cluster_3_df(self, column, df):
        return (df[column] - df[column].mean()) / df[column].std()

    def calc_distance(self, vector_one, vector_two, distance_metric='manhattan'):
        if distance_metric == 'manhattan':
            return self.calc_manhattan(vector_one, vector_two)
        elif distance_metric == 'euclidean':
            return np.linalg.norm(vector_one - vector_two)

    def calc_manhattan(self, vector_one, vector_two):
        distance = 0
        for i in range(len(vector_one)):
            distance = distance + (vector_one[i] - vector_two[i]) if distance + (vector_one[i] - vector_two[i]) >= 0 else -1 * (distance + (vector_one[i] - vector_two[i]))
        return float(distance)

    def score_function(self):
        count = 0.0
        for i in range(len(self.centroids)):
            for j in self.class_labels[i]:
                count = count + math.pow(self.calc_distance(j, self.centroids[i], self.d_type), 2)
        return count

    def predict(self, df, k, option):

        self.centroids = np.array([df[x] for x in np.random.choice(len(df), k)])

        if option == 5:
            df = df[np.random.choice(df.shape[0], int(len(df) * 0.06), replace=False), :]

        opt_dist = 0.0

        while True:
            overall_dist = 0.0

            for a in range(k):
                self.class_labels[a] = []

            for value in df:
                distance = [self.calc_distance(value, cen, self.d_type) for cen in self.centroids]
                overall_dist = (overall_dist + min(distance))
                self.class_labels[distance.index(min(distance))].append(value)

            for b in self.class_labels:
                if len(self.class_labels[b]) > 0:
                    self.centroids[b] = np.array(self.class_labels[b]).mean(axis=0)

            if overall_dist - opt_dist < 0:
                total_dist = -1 * (overall_dist - opt_dist)
            else:
                total_dist = overall_dist - opt_dist

            if total_dist >= 0.0001:
                opt_dist = overall_dist
            else:
                break

        return self.centroids


if __name__ == "__main__":
    data_path = '../data/given/'
    # data_name = sys.argv[1]
    # k = int(sys.argv[2])
    # option = int(sys.argv[3])

    # if len(sys.argv) != 4:
    #     print('INVALID')
    #     sys.exit(0)

    data_name = 'yelp3.csv'
    count = 1
    k = 20
    option = 6
    # k = int(sys.argv[2])
    # option = int(sys.argv[3])

    k_means = KMeans(option)
    data_frame = k_means.input_stream(data_path + data_name, option)

    # centroidVals = k_means.predict(data_frame, int(sys.argv[2]), int(sys.argv[3]))
    centroidVals = k_means.predict(data_frame, k, option)

    print("WC-SSE=" + str(k_means.score_function()))

    for index in range(0, len(centroidVals)):
        print("Centroid", str(index+1), centroidVals[index])