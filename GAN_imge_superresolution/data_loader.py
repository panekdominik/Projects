from skimage.transform import resize
from skimage import io
import glob
import numpy as np

def data_loader(lr_path, hr_path, img_width, img_height):
    
    lr_images = []
    lr_image_data = sorted(glob.glob(lr_path+"/*"),
        key=lambda x: int(x.split("fluo_image_")[-1].split(".")[0]))
    
    for img in lr_image_data:
        image = io.imread(img)
        image = resize(image, (img_width, img_height, 3), mode = 'constant', preserve_range = True)
        lr_images.append(image)
    lr_images = np.array(lr_images)

    hr_images = []
    hr_image_data = sorted(glob.glob(hr_path+"/*"),
        key=lambda x: int(x.split("conf_image_")[-1].split(".")[0]))
    for img in hr_image_data:
        image = io.imread(img)
        image = resize(image, (img_width, img_height, 3), mode = 'constant', preserve_range = True)
        hr_images.append(image)
    hr_images = np.array(hr_images)

    return lr_images, hr_images