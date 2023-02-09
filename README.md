# image_segmenttion
Image segmentation codes, including contouring, maksing, object finding, windowing, denoising etc. 
Packages required for this code to work: numpy, matplotlib.pyplot, seaborn, os, skimage, scipy. 

The fisrst part of this code (still under development) takes as an input grayscale tomographic (micro-CT) images, and acts on them returning the contours, masked objects (organs and vessels), areas/volumes of the objects of interests etc. 
The second part of this code (still under development) is supposed to take as an input masks (found in the first part), label them with integer numbers and use machine learning models (the ones with "fit", and "predict" methods) and tries to identify the objects visible at the tomographic images. 
