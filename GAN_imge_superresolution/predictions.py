from keras.models import load_model
from cycler import cycler
from prediction_helpers import image_loader, plot_samples, plot_corss_section, calculate_metrics
from data_loader import data_loader
from csv_saver import convert_tensor_events_to_csv, return_tags
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
parser.add_argument("--model_path", type=str, help="Path to the model file (make sure model is compatible with data)")
parser.add_argument("--pixel_number", type=int, default=-1, help="Number indicating the slice in y direction across which the plot will be made")
parser.add_argument("--tensorflow_data", type=str, default=None, help="Path to a tensorflow file sontaining losses saved during training.")
parser.add_argument("--log", type=str, default=None, help="Name of the moels stored ine the tensorflow log. These names may be necessary to recover csv version of tensorflow file.")
parser.add_argument("--csv", type=str, default=None, help="Name of the csv file to be created from the tensorflow log")
parser.add_argument("--disc_real_loss", type=str, default=None, help="Name of the discriminator real loss value in the log to be saved in csv file. If you dont know the name of the moddel loss stored in the tensorflow log run --logs command")
parser.add_argument("--disc_fake_loss", type=str, default=None, help="Name of the discriminator fake loss value in the log to be saved in csv file")
parser.add_argument("--gen_loss", type=str, default=None, help="Name of the generator value in the log to be saved in csv file")

args = parser.parse_args()
lr_dir = args.lr_dir
hr_dir = args.hr_dir
saving_dir = args.saving_dir
img_width = args.image_width
img_height = args.image_height
random_sample = args.random
model_path = args.model_path
pixel = args.pixel_number
tensorflow_data = args.tensorflow_data
logs = args.log
csv = args.csv
disc_real_loss = args.disc_real_loss
disc_fake_loss = args.disc_fake_loss
gen_loss = args.gen_loss

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

# Get names of the logs in tensorflow 
if logs != None:
    return_tags(tensor_data=logs)

# Saving model losses to csv file 
if tensorflow_data != None:
    if disc_real_loss != None:
        convert_tensor_events_to_csv(log_dir=tensorflow_data, tensor_tag=disc_real_loss, csv_output_filename=csv, header=disc_real_loss)
    if disc_fake_loss != None:
        convert_tensor_events_to_csv(log_dir=tensorflow_data, tensor_tag=disc_fake_loss, csv_output_filename=csv, header=disc_fake_loss)
    if gen_loss != None:
        convert_tensor_events_to_csv(log_dir=tensorflow_data, tensor_tag=gen_loss, csv_output_filename=csv, header=gen_loss)