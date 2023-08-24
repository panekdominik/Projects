from numpy.random import randint
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error, normalized_root_mse, peak_signal_noise_ratio, structural_similarity

def image_loader(random_sample, hr_images, lr_images):
    if random_sample == -1:
        ix = randint(0, len(lr_images)+1, 1)
        conf_image = hr_images[ix]
        fluo_image = lr_images[ix]
        print("Image number:", ix)
    else:
        conf_image = hr_images[[random_sample]]
        fluo_image = lr_images[[random_sample]]
        print("Image number:", random_sample)

    return fluo_image, conf_image

def plot_samples(hr_image, lr_image, model, saving_dir):
    
    fig, ax = plt.subplots(1, 3, figsize=(16,5))
    ax[0].imshow(hr_image[0])
    ax[0].set_title("High-resolution Image", fontsize=15)
    ax[1].imshow(lr_image[0])
    ax[1].set_title("Low-resolution image", fontsize=15)
    ax[2].imshow(model[0])
    ax[2].set_title("Generated image", fontsize=15)
    plt.tight_layout()
    plt.savefig(saving_dir+'/generated_images.png', dpi=600)
    plt.show()

def plot_corss_section(hr_image, lr_image, model, saving_dir, pixel):

    generated = np.clip(model[0], a_min=0, a_max=1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    ax[0].plot(hr_image[0][:, 128, :])
    ax[0].set_title("High-resolution profile");
    ax[1].plot(lr_image[0][:, 128, :])
    ax[1].set_title("Low-resolution profile")
    ax[2].plot(generated[:, 128, :])
    ax[2].set_title("Network profile")

    ax[1].set_xlabel('Pixel number')
    ax[0].set_ylabel('Normalized intensity')
    plt.tight_layout()
    plt.savefig(saving_dir+'/cross_section_{}.png'.format(pixel), dpi=600, transparent=True)
    plt.show()

def calculate_metrics(hr_image, model):
    
    mse_hr_gen = mean_squared_error(hr_image, model)
    nrmse_hr_gen = normalized_root_mse(hr_image, model)
    psnr_hr_gen = peak_signal_noise_ratio(hr_image, model)
    ssim_hr_gen = structural_similarity(hr_image[0], model[0], multichannel=True)

    print("Metrics between high-resoluiton and generated image:", '\n', 
          "MSE: ", mse_hr_gen, '\n',
          "NRMSE: ", nrmse_hr_gen, '\n',
          "PSNR: ", psnr_hr_gen, '\n',
          "SSIM: ", ssim_hr_gen)