#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import _pickle as pkl
import sys
from tqdm import tqdm
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} event_name feat_dir feat_dim output_file").format(sys.argv[0])
        print ("feat_dir -- dir of feature files")
        print ("feat_dim -- dim of features")
        print ("output_file -- path to save the svm model")
        exit(1)

    feat_dir = sys.argv[1]
    feat_dim = int(sys.argv[2])
    output_file = sys.argv[3]

    train_files = open("../all_trn.lst","r").readlines()

    labels = []
    features = []

    print("Starting training of Kaggle model")
    for idx,line in enumerate(tqdm(train_files)):
        fname, label = line.strip().split()
        feature_file = feat_dir + fname + ".csv"
        if os.path.exists(feature_file) == False:
            continue
        else:
            features.append(np.loadtxt(feature_file,delimiter=";"))
            if label in ["P001","P002","P003"]:
                labels.append(int(label[-1]))
            else:
                labels.append(0)

    features, labels = np.array(features), np.array(labels)
    pos_ex = features[labels!=0]
    neg_ex = features[labels==0]
    pos_labels = labels[labels!=0]

    total_pos = len(pos_ex)
    np.random.shuffle(neg_ex)

    neg_ex = neg_ex[:total_pos]
    new_features = np.vstack([pos_ex,neg_ex])
    new_labels = list(pos_labels) + ([0]*total_pos)
    # svm = SVC(kernel='linear', probability=True)
    # model = GaussianNB()
    model = RandomForestClassifier(n_estimators=100,max_depth=5)
    model.fit(new_features,new_labels)
    
    with open(output_file,"wb") as o:
        pkl.dump(model,o)

    print ('Kaggle Model trained successfully for Kaggle!')
