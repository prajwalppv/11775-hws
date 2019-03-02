Run the below command in the bash to setup folders in the hw2_code folder: <br>
    ` mkdir -p cnn surf cnn_2 kmeans kmeans_cnn cnn_pred surf_pred models downsampled_videos `
<br>

To downsample videos:<br>
* Start the notebook - Downsample.ipynb <br>
* Run the cells in sequence after changing the paths as required <br>
   
Change parameters as required in the config.yaml file <br>

To extract SURF features run: <br>
` python surf_feat_extraction.py list/all.video config.yaml `
    
To extract CNN features: <br>
* Start the notebook - CNN_features.ipynb <br>
* Change the variables to indicate the right video files and destination path and run the cells sequentially <br>
    
Once features are extracted: <br>
* Open the notebook, surf.ipynb <br>
* Specify the paramters such as MODE='surf'/'cnn' to select which features you want to use. <br>
* Use the "Run All Cells" option/ run cells sequentially
