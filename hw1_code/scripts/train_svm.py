#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import _pickle as pkl
import sys
from tqdm import tqdm

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print ("Usage: {0} event_name feat_dir feat_dim output_file").format(sys.argv[0])
        print ("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print ("feat_dir -- dir of feature files")
        print ("feat_dim -- dim of features")
        print ("output_file -- path to save the svm model")
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    train_files = open("../all_trn.lst","r")
    val_files = open("../all_val.lst")

    labels = []
    features = []

    print("Starting training of SVM model")
    for line in tqdm(train_files):
        fname, label = line.strip().split()
        feature_file = feat_dir + fname + ".csv"
        if os.path.exists(feature_file) == None:
            continue
        else:
            features.append(np.loadtxt(feature_file,delimiter=";"))
            if label == event_name:
                labels.append(1)
            else:
                labels.append(0)
    features, labels = np.array(features), np.array(labels)

    svm = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svm.fit(features,labels)

    with open(output_file,"wb") as o:
        pkl.dump(svm,o)
        
    print ('SVM trained successfully for event %s!' % (event_name))
