from keras.models import load_model
from cycler import cycler
from prediction_helpers import image_loader, plot_samples, plot_corss_section, calculate_metrics
from data_loader import data_loader
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="Predictions",
                                 description="This script allows to see the results of a trained GAN model. It produces a plot showing three images, HR, LR and generated (based on LR). It also shows the corss-section through given image slice allowing to compare detailedness. It also shows the metrics like MSE, NRMSE, PSNR, and SSIM allowing to comapre HR and generated images quantitively.",
                                 epilog="To learn more go to: https://github.com/panekdominik/PhD_projects/tree/main/GAN_imge_superresolution or read README file.")
parser.add_argument("--lr_dir", type=str, help="Directory where the low-resolution images are stored")
parser.add_argument("--hr_dir", type=str, help="Directory where the high-resolution images are stored")
parser.add_argument("--saving_dir", type=str, help="Directory where the image will be saved")
parser.add_argument("--image_width", type=int, help="Width of the image")
parser.add_argument("--image_height", type=int, help="Height of the image")
parser.add_argument("--random", type=int, default=-1, help="If no integer is specified random image is taken if number is specified image with given id is taken")
parser.add_argument("--model_path", type=str, help="Path to the model file (make super model is compatible with data)")
parser.add_argument("--pixel_number", type=int, default=-1, help="Number indicating the slice in y direction across which the plot will be made")

args = parser.parse_args()
lr_dir = args.lr_dir
hr_dir = args.hr_dir
saving_dir = args.saving_dir
img_width = args.image_width
img_height = args.image_height
random_sample = args.random
model_path = args.model_path
pixel = args.pixel_number

plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

# Loading data and choosing random or specified image
fluo_images, conf_images = data_loader(lr_path=lr_dir, hr_path=hr_dir, img_width=img_width, img_height=img_height)
fluo_image, conf_image = image_loader(random_sample=random_sample, hr_images=conf_images, lr_images=fluo_images)

# Loading model and predicting image
GAN_model = load_model(model_path, compile=False)
GAN = GAN_model.predict(fluo_image)

# Plotting exemplary samples
plot_samples(hr_image=conf_image, lr_image=fluo_image, model=GAN, saving_dir=saving_dir, random_sample=random_sample)

# Plotting cross section along y-axis through the images
plot_corss_section(hr_image=conf_image, lr_image=fluo_image, model=GAN, saving_dir=saving_dir, pixel=pixel, random_sample=random_sample)

# Calculating some metrics between hr and generated imge
calculate_metrics(hr_image=conf_image, model=GAN)