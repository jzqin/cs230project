# Using CNNs To Detect and Segment Aneurysms

*Authors: Jason Qin and Harry Emeric*

This repo contains the code to preprocess the Medical Imaging data, and use it to train models forked from the CSAIL Semantic Segmentation repository (https://github.com/CSAILVision/semantic-segmentation-pytorch).


### NOTE: unfortunately, because the data are medically classified, the results within the Jupyter notebooks cannot be reproduced from just the repository

## Preprocessing
Preprocessing converts CT scans stored in DICOM format and segmentation masks stored in NRRD format to PNG. Next, preprocessing generates a mapping from the CT scan PNGs to the segmentation PNGs, in the format necessary for the semantic segmentation networks developed by CSAIL.

The preprocessing code is located in the following files:
* ./cta-scripts/organize_and_preprocess_data.ipynb : high-level script showing file processing  
* ./cta-scripts/data_augmentation.ipynb : determines which PNG mask and CT scan images contain aneurysms; upsamples these positive aneurysm images  
* ./cta-scripts/preprocess_data.py : code for actually converting from DICOM and NRRD to PNG, as well as organizing data as input for NN  
* ./cta-scripts/util/* : utility functions for preprocess_data.py  

## Modelling
Modeling includes testing different 1) architectures, 2) upsampling extents, 3) loss weights, 4) learning rate  
* ./cta-scripts/architecture_and_hyperparam_search.ipynb : script to automate invoking calling the CSAIL network via shell commands  
* ./cta-scripts/evaluate_models.ipynb : after a model is trained, this script runs the model on the test data  
* ./resnet/semantic-segmentation-pytorch : code for repository used for modeling

We changed the following aspects of the original semantic-segmentation repo:  
* train.py :
** added weights to negative log likelihood function  
** model originally ignores background pixels in loss function, so we changed the data preprocessing in the code to include background pixel error  
* data/ : changed class labels for our 2 class segmentation  

## Analysis
Analyze data by seeing how training loss and IOU change over time, visualizing data with plots  
* ./cta-scripts/visualize_metrics.ipynb : code for visualizing training IOU and loss for architecture and hyperparameter tuning  
* ./cta-scripts/visualize_data-Error_analysis.ipynb : visualize individual images for determining model deficits and prediction capabilities  



