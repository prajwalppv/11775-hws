#!/bin/python 

import numpy
import os
from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster import MiniBatchKMeans
import _pickle as pkl
import sys


def getMiniBatches(file, batch_size):
    with open(file) as f:
        while True:
            line = f.readline()
            batch = ""
            if line:
                batch += line
            i = 1
            while line != None and i<batch_size:
                line = f.readline()
                if not line:
                    break
                batch += line
                i += 1
            print(batch)
            array = numpy.loadtxt(batch,delimiter=";")
            yield array
        

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
    input_mfccs = numpy.loadtxt(mfcc_csv_file,delimiter=";")    # Doesnt work -> Memory error

    # Initialize KMeans model with required parameters
    # data = getMiniBatches(mfcc_csv_file,batch_size)
    # for i,r in enumerate(data):
    #     print(i,r)
    kmeans = MiniBatchKMeans(n_clusters=cluster_num,verbose=True)
    # Fit the model to the data
    kmeans.fit(input_mfccs)
    # Save the trained model using Pickle
    with open(output_file,"wb") as out:
        pkl.dump(kmeans,out)

    print ("K-means trained successfully!")
