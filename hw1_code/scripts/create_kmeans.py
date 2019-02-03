#!/bin/python
import numpy as np
import os
import _pickle as pkl
from sklearn.cluster.k_means_ import KMeans
import sys
from collections import Counter
from tqdm import tqdm
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} kmeans_model, cluster_num, file_list").format(sys.argv[0])
        print ("kmeans_model -- path to the kmeans model")
        print ("cluster_num -- number of cluster")
        print ("file_list -- the list of videos")
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    fread = open(file_list,"r")
    filenames = fread.readlines()

    all_video_features = [None]*len(filenames)
    # load the kmeans model
    kmeans = pkl.load(open(kmeans_model,"rb"))
    for idx,line in enumerate(tqdm(filenames)):
        bag_of_words_feature = np.zeros(shape=(cluster_num))
        mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        feature_path = "feature/kmeans/" + line.replace('\n','') + ".csv"
        if os.path.exists(mfcc_path) == False:
            continue
        mfcc_input = np.loadtxt(mfcc_path, delimiter=";")
        cluster_pred = kmeans.predict(mfcc_input)
        histogram = Counter(cluster_pred)
        for k,v in histogram.items():
            bag_of_words_feature[k] = v

        np.savetxt(feature_path,bag_of_words_feature,delimiter=";")
    print ("K-means features generated successfully!")
