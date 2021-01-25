# Automated_Nerves_Detection
An automated nerves detection implementation with a deep learning based approach.  

Architecture based on Keras.

The code is in jupyter notebook. 
It consists of step-by-step guide/tutorial for an automated nerves detection implementation on immunohistochemistry specimens of thyroid cancer. 

# Getting Started
1. Download the repository.
2. Ensure that the dependencies are installed. 
    - Tensorflow >= 1.12.0
    - Keras >= 2.2.4
    - Openslide >= 1.1.1
    - Sklearn >= 0.19.1
    - Cv2 >= 3.4.4
    - Numpy >= 1.15.4
    - Matplotlib >= 2.2.2
    - Skimage >= 0.13.1
    - PIL >= 5.1.0
    - Scipy >= 1.1.0
    - Csv >= 1.0
    
3. Download the svs files (e.g. xxxxx.svs) and place it in dataFiles>svs files folder.
4. Ensure that the annotation files (e.g. AnnotFile xxxxx.csv) are in dataFiles>"annot csv" folder.
5. Ensure that the pretrained model (e.g. pretrained_DL-B.h5) is in dataFiles>"pretrained models" folder.
6. Ensure that sArea.csv and sArea_Demo.csv files are in dataFiles folder.
5. Run the jupyter notebook.

# Acknowledgements
Keras: Keras.io
