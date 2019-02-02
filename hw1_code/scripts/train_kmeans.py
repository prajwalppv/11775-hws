#!/bin/python 

import numpy
import os
from sklearn.cluster.k_means_ import KMeans
import _pickle as pkl
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} mfcc_csv_file cluster_num output_file").format(sys.argv[0])
        print ("mfcc_csv_file -- path to the mfcc csv file")
        print ("cluster_num -- number of cluster")
        print ("output_file -- path to save the k-means model")
        exit(1)

    mfcc_csv_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])

    #Read input MFCCS
    input_mfccs = numpy.genfromtxt(mfcc_csv_file,delimiter=";")
    # Initialize KMeans model with required parameters
    kmeans = KMeans(n_clusters=cluster_num,verbose=True)
    # Fit the model to the data
    kmeans.fit(input_mfccs)
    # Save the trained model using Pickle
    with open(output_file,"wb") as out:
        pkl.dump(kmeans,out)

    print ("K-means trained successfully!")
