from loss_functions import ssim_loss
from tensorflow.keras.losses import binary_crossentropy
from skimage.metrics import mean_squared_error
import tensorflow as tf


def calculate_validation_loss(gen_model, disc_model, val_data):
    val_images_real, val_images_fake = val_data

    # Generate fake images
    val_images_generated = gen_model.predict(val_images_fake)

    # Calculate discriminator loss (BCE)
    disc_loss_real = binary_crossentropy(tf.ones_like(disc_model(val_images_real)), disc_model(val_images_real))
    disc_loss_fake = binary_crossentropy(tf.zeros_like(disc_model(val_images_generated)), disc_model(val_images_generated))
    val_disc_loss = tf.reduce_mean(disc_loss_real + disc_loss_fake)

    # Calculate generator loss (MSE, BCE, SSIM)
    val_gen_mse_loss = mean_squared_error(val_images_real, val_images_generated)
    val_gen_bce_loss = binary_crossentropy(tf.ones_like(disc_model(val_images_generated)), disc_model(val_images_generated))
    val_gen_ssim_loss = ssim_loss(y_true=val_images_real, y_pred=val_images_generated)
    val_gen_loss = val_gen_bce_loss+0.1*val_gen_ssim_loss+0.01*val_gen_mse_loss

    # Total validation loss
    total_val_loss = val_disc_loss + val_gen_loss

    return total_val_loss.numpy()