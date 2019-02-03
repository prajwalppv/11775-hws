#!/bin/python 

import numpy
import os
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import MiniBatchKMeans
import _pickle as pkl
import sys
from time import time

# Performs K-means clustering and save the model to a local file
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print ("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print ("mfcc_csv_file -- path to the mfcc csv file")
        print ("cluster_num -- number of cluster")
        print ("output_file -- path to save the k-means model")
        print ("mini-batch size -- Size of mini-batch")
        exit(1)

    mfcc_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    batch_size = int(sys.argv[4])

    #Read input MFCCS
    print("Reading MFCCs into numpy array")
    start = time()
    input_mfccs = numpy.loadtxt(mfcc_csv_file,delimiter=";")    # Doesnt work -> Memory error
    end = time()
    print("Loaded input into memory. Time taken: {}".format(end-start))
    # Initialize KMeans model with required parameters
    kmeans = MiniBatchKMeans(n_clusters=cluster_num,batch_size=batch_size,verbose=True)

    # Fit the model to the data
    print("Fitting K-means model")
    start = time()
    kmeans.fit(input_mfccs)
    end = time()
    print("K-means fit complete. Time taken: {}".format(end-start))
    # Save the trained model using Pickle
    with open(output_file,"wb") as out:
        pkl.dump(kmeans,out)

    print ("K-means trained successfully!")
