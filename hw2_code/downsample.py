from joblib import Parallel,delayed
from tqdm import tqdm
import os
# ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
downsampling_frame_len = 60
downsampling_frame_rate=15
video_path = "~/video/"
output_directory = "downsampled_videos/"

def downsample(filename):
    cmd = "ffmpeg -y -ss 0 -i {}.mp4 -strict experimental -t {} -r {} {}.mp4".format(video_path+filename , 
                                                                                    downsampling_frame_len,
                                                                                    downsampling_frame_rate,
                                                                                    output_directory+filename)
    
    os.system(cmd)
    
with Parallel(n_jobs=16) as parallel:
    all_files = open("list/all.video").readlines()
    all_files = [file.strip() for file in all_files]
#     print(all_files)
    res = parallel(delayed(downsample)(file) for file in all_files)