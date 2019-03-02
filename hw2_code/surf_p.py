#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle as pkl
import pdb
from tqdm import tqdm
from joblib import Parallel,delayed


def get_surf_features_from_video(surf,downsampled_video_filename, surf_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    # TODO
    downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.mp4')
#         print(downsampled_video_filename)
    surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')
    frames = get_keyframes(downsampled_video_filename,keyframe_interval)
    features = []
    max_len = 0
    for keyframe in frames:
        key, des = surf.detectAndCompute(keyframe,None)
        features.append(des)

    with open(surf_feat_video_filename,"wb") as o:
        pkl.dump(features,o)
        

def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)
        
    print("Beginning SURF extraction")

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object
    surf = cv2.SURF(hessian_threshold)

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    videos = [i.strip() for i in fread.readlines()]
    with Parallel(n_jobs=4) as parallel:
        res = parallel(delayed(get_surf_features_from_video)(surf,video_name,keyframe_interval) for video_name in tqdm(fread.readlines()))
