import glob 
import os
import numpy as np
import cv2
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder

def read_data_dir(directories, prefix = "Maska_"):
    
    data = []
    for directory in directories:
        folders = sorted(os.listdir(directory))
        for folder in folders:
            time_series = sorted(glob.glob(directory+'/{}/*'.format(folder)),
                                 key=lambda x: int(x.split(prefix)[-1].split(".")[0]))
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
    image_data = np.array(image_data)
    return image_data

def labels_reader(path, seq_length=161, num_seq=48):

    labels = np.loadtxt(path)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = labels_encoded.reshape((seq_length, num_seq))

    updated_cell_types=[]
    for seq in labels_encoded:
        for idx, state in enumerate(seq):
            if state == 0:
                updated_cell_types.append(0)
            elif state == 1:
                updated_cell_types.append(1)
            elif state == 2:
                updated_cell_types.append(2)
            elif state == 3:
                updated_cell_types.append(2)
            elif state == 4:
                if seq[idx-1] == 0:
                    updated_cell_types.append(0)
                elif seq[idx-1] == 1:
                    updated_cell_types.append(1)
                else:
                    updated_cell_types.append(2)
    
    updated_cell_types = np.array(updated_cell_types)
    n_classes = len(np.unique(labels))

    return updated_cell_types, n_classes

def data_windowing(data, window_size = 3):

    sequence_length = data.shape[0]
    input_data = []
    for i in range(sequence_length - window_size + 1):
        subset = data[i:i+window_size]
        input_data.append(subset)

    windowed_data = np.array(input_data)

    return windowed_data