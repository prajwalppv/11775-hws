#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import _pickle as pkl
import sys
from tqdm import tqdm

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print ("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print ("model_file -- path of the trained svm file")
        print ("feat_dir -- dir of feature files")
        print ("feat_dim -- dim of features; provided just for debugging")
        print ("test_output_file -- path to save the prediction score for test dataset")
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    test_output_file = sys.argv[4]
    f_write = open(test_output_file,"w")


    # Set path to val and test lists    
    test_files = open("../all_test.video").readlines()
    
    # Load SVM model
    model = None
    with open(model_file,"rb") as o:
        model = pkl.load(o)
    # Generate TEST probabilities
    print("Creating TEST predictions")
    for line in tqdm(test_files):
        fname = line.strip()
        feature_file = feat_dir + fname + ".csv"
        if os.path.exists(feature_file) == False:
            f_write.write("{},{}\n".format(fname,0))
            continue
        else:
            X = np.loadtxt(feature_file,delimiter=";")
            X = np.reshape(X,(1,-1))
            class_label = model.predict(X)
            f_write.write("{},{}\n".format(fname,class_label))