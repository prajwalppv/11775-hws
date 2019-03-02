Run the below command in the bash to setup folders in the hw2_code folder:
    mkdir -p cnn surf cnn_2 kmeans kmeans_cnn cnn_pred surf_pred models downsampled_videos

To downsample videos 
    - Start the notebook - Downsample.ipynb
    - Run the cells in sequence after changing the paths as required
    
Change parameters as required in the config.yaml file

To extract SURF features run:
    python surf_feat_extraction.py all_video_file_list config_file
    
To extract CNN features,
    - Start the notebook - CNN_features.ipynb
    - Change the variables to indicate the right video files and destination path and run the cells sequentially
    
Once features are extracted,
    - Open the notebook, surf.ipynb
    - Specify the paramters such as MODE='surf'/'cnn' to 
