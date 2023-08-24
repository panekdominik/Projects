from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from datetime import datetime
from helpers import generate_real_samples, generate_fake_samples, generate_validation_samples
from validation import calculate_validation_loss
import argparse
from models import define_discriminator, define_generator, define_gan
from data_loader import data_loader

parser = argparse.ArgumentParser()
parser.add_argument("--lr_dir", type=str, help="Directory where the low-resolution images are stored")
parser.add_argument("--hr_dir", type=str, help="Directory where the high-resolution images are stored")
parser.add_argument("--saving_dir", type=str, help="Directory where the model will be saved")
parser.add_argument("--image_width", type=int, help="Width of the image")
parser.add_argument("--image_height", type=int, help="Height of the image")
parser.add_argument("--epochs", type=int, default=10000, help="Number of iterations the model will run for")
parser.add_argument("--BCE_weight", type=float, help="Weigths given to the BCE loss")
parser.add_argument("--MSE_weight", type=float, help="Weigths given to the MSE loss)")
parser.add_argument("--SSIM_weight", type=float, help="Weigths given to the SSIM loss")
parser.add_argument("--model_weights", type=str, default=None, help="Model weigths imported from .h5 file")

args = parser.parse_args()
lr_dir = args.lr_dir
hr_dir = args.hr_dir
saving_dir = args.saving_dir
img_width = args.image_width
img_height = args.image_height
epochs = args.epochs
BCE_loss = args.BCE_weight
MSE_loss = args.MSE_weight
SSIM_loss = args.SSIM_weight
model_weights = args.model_weights

def train_GAN(d_model, g_model, gan_model, data, save_path, epochs = 300, n_batch = 1, model_weights=None):
    
    n_patch = d_model.output_shape[1]

    conf_images, _ = data

    batches_per_epoch = int(len(conf_images)/n_batch)

    n_steps = batches_per_epoch * epochs
    best_val_loss = float('inf')

    lr_controller_g = ReduceLROnPlateau(model=gan_model, factor=0.5, patience=10, mode='min', min_delta=1e-4,
                                    cooldown=0, min_lr=1e-4 * 0.1, verbose=1)
    lr_controller_d = ReduceLROnPlateau(model=d_model, factor=0.5, patience=10, mode='min', min_delta=1e-4,
                                    cooldown=0, min_lr=1e-5 * 0.1, verbose=1)
    lr_controller_g.on_epoch_begin(epoch=1)
    lr_controller_d.on_epoch_begin(epoch=1)

    if model_weights:
        g_model.load_weights(model_weights)

    summary_writer = tf.summary.create_file_writer(save_path + 'logs')

    start1 = datetime.now()

    for i in range(n_steps):
        [conf_img, fluo_img], y_real = generate_real_samples(data, n_batch, n_patch)
        pred, y_fake = generate_fake_samples(g_model, fluo_img, n_patch)

        disc_loss1 = d_model.train_on_batch(conf_img, y_real.reshape(-1, 1))
        disc_loss2 = d_model.train_on_batch(pred, y_fake.reshape(-1, 1))
        gen_loss, _, _ = gan_model.train_on_batch(fluo_img, [y_real.reshape(-1, 1), conf_img])

        # Logging
        if i % 100==0:
            with summary_writer.as_default():
                tf.summary.scalar('Discriminator_Loss1', disc_loss1[0], step=i)
                tf.summary.scalar('Discriminator_Loss2', disc_loss2[0], step=i)
                tf.summary.scalar('Generator_Loss', gen_loss, step=i)

        if (i+1)%2000== 0:
            g_model.save(save_path+'UNET_GAN_model6_Adv1_MSE001_SSIM01_epoch_{}.h5'.format(i))

        val_data = generate_validation_samples(data, n_samples=8)
        val_losses = calculate_validation_loss(g_model, d_model, val_data)
        val_loss = val_losses.mean()
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          g_model.save(save_path + 'best_model_{}.h5'.format(i))

        print('Step: {}, Discriminator losses: D1 {}, D2 {}, Generator loss: G {}'.format(i+1,
                                                                                          round(disc_loss1[0], 8),
                                                                                          round(disc_loss2[0], 8),
                                                                                          round(gen_loss, 8)))
        print('Best Validation Loss:', best_val_loss)

    # Save the last model after training
    g_model.save(save_path + 'last_model.h5')
    
    stop1 = datetime.now()
    execution_time = stop1-start1
    print("Execution time is: ", execution_time)

    return best_val_loss

### load data
fluo_images, conf_images = data_loader(lr_path=lr_dir, hr_path=hr_dir, img_width=img_width, img_height=img_height)
img_shape = fluo_images.shape[1:]
data = [conf_images, fluo_images]
loss_weights = [BCE_loss, MSE_loss, SSIM_loss]

print(img_shape)

### define models
d_model = define_discriminator(image_shape=img_shape)
g_model = define_generator(image_shape=img_shape)
gan_model = define_gan(gen_model=g_model, disc_model=d_model, image_shape=img_shape, weights=loss_weights)


train_GAN(d_model=d_model, 
          g_model=g_model, 
          gan_model=gan_model, 
          save_path=saving_dir, 
          data=data, 
          epochs=epochs, 
          n_batch=1, 
          model_weights=model_weights)