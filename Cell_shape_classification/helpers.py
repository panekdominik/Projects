import glob 
import os
import numpy as np
import cv2
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def read_data_dir(directories):
    
    data = []
    for directory in directories:
        folders = sorted(os.listdir(directory))
        for folder in folders:
            time_series = sorted(glob.glob(directory+'/{}/*'.format(folder)),
                                 key=lambda x: int(x.split("Klatka_")[-1].split(".")[0]))
            data.append(time_series)
    data = np.array(data).flatten()
    
    return data

def img_reader(data_files, image_width, image_height, n_channels):
    
    image_data = []
    for chunk in data_files:
        img = cv2.imread(chunk)
        img = resize(img, (image_height, image_width, n_channels), preserve_range=True)
        img = img/255
        img = (img>0.5).astype(np.uint8)
        image_data.append(img)
    
    return image_data

def labels_reader(path):

    labels = np.loadtxt(path)
    n_classes = len(np.unique(labels))
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels = to_categorical(labels_encoded, num_classes=n_classes)
    labels = labels.reshape((labels.shape[0], n_classes))

    return labels, n_classes