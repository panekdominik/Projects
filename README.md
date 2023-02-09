# image_segmenttion_and_classification
Image segmentation codes, including contouring, maksing, object finding, windowing, denoising etc. 
Packages required for this code to work: **numpy, matplotlib.pyplot, seaborn, os, skimage, scipy**. 

The first part of this code (still under development) takes as an input grayscale tomographic (micro-CT) images, and acts on them returning the contours, masked objects (organs and vessels), areas/volumes of the objects of interests etc. 
The second part of this code (still under development) is supposed to take as an input masks (found in the first part), label them with integer numbers and use machine learning models (the ones with "fit", and "predict" methods) and tries to identify the objects visible at the tomographic images. 

Functions used for segmentation:
1) **image_show(image, nrows=1, ncols=1, cmap='gray', kwargs)** takes an image as an input and shows it in the grayscale 
2) **window_image(image, window_center, window_width)** takes an image as an input and returns the windowed images (based on the window center and width)
3) **remove_noise(image, window_center, window_width)** takes an image as an input and returns denised image (image withoud single pixel speckles)
4) **contour_distance(contour)** takes a list of contours as an input and checks the Euclidian distance between the first and the last point of a given contour
5) **set_is_closed(contour)** takes a list of contours and checks if the contour is closed (based on the information from the Euclidian distances)
6) **find_object(contours, volume_size)** takes the countours and volume_size as arguments, and returns the contour of a given object (contour is found based on the volume (in 2D case area) of the object)
7) **create_mask_from_polygon** takes as inputs image and countours, and returns the mask of the object (based on the contour) - binary array where 1 represent object and 0 the rest
8) **create_vessel_mask(liver_mask, ct_numpy)** takes as inputs mask and an original image, it returns the mask for the objects which were not selected earlier by the contour.
9) **segmentation(path, window_center=60, window_width=5, contour_lenght = 5, volume = 50000)** cumulative function collecting all the functions described above in one place. It takes as an argument path to the directory where the images are etc. It returns masked objects, masked vessels, and all contours. 

Function sused for classification:
1) ??
2) ??
3) ??
