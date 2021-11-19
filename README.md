# Automated Nerves Quantification
An automated nerves quantification implementation with a deep learning segmentation model.  

Architecture based on Keras.

The code is in jupyter notebook. 
It consists of step-by-step guide/tutorial for nerves segmentation approach on immunohistochemistry specimens of thyroid cancer. 

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
    
3. Download the svs files (e.g. xxxxx.svs) from https://doi.org/10.5281/zenodo.4459863 and place it in dataFiles>"svs files" folder.
4. Ensure that the annotation files (e.g. AnnotFile xxxxx.csv) are in dataFiles>"annot csv" folder.
5. Ensure that the pretrained model (e.g. pretrained_DL-B.h5) is in dataFiles>"pretrained models" folder.
6. Ensure that sArea.csv and sArea_Demo.csv files are in dataFiles folder.
7. Run the jupyter notebook (i.e. Nerves Segmentation Demo.ipynb).
8. For visualisation of the complete results, 
   - download extra>DrawingPredictionLabels.groovy
   - ensure the file path to the prediction file "PredFile xxxxx.csv" is correct
   - run the script on QuPath, download here https://qupath.github.io/, under the menu bar find the groovy script in Automate>Project scripts.
9. For direct deep learning approach, replace all the python files in the main folder with the python files in the Direct_Deep_Learning folder as there are some changes in path and file names that is made in the associated functions for the direct deep learning script to run appropriately.

# Acknowledgements
Keras: Keras.io
