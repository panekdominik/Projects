from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, AveragePooling2D, LeakyReLU, Concatenate, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Dense, Input, add, Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau
from skimage.metrics import mean_squared_error
from skimage.transform import resize
from skimage import io
import tensorflow as tf
import numpy as np
from datetime import datetime
import glob

def disc_block(inputs, channels):

    conv1 = Conv2D(channels, (4, 4), strides=(2, 2), padding='same')(inputs)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv2 = Conv2D(channels, (4, 4), strides=(2, 2), padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.1)(conv2)

    conv3 = Conv2D(channels, (4, 4), strides=(2, 2), padding='same')(conv2)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv4 = Conv2D(channels, (4, 4), strides=(2, 2), padding='same')(conv3)
    conv4 = LeakyReLU(alpha=0.1)(conv4)

    return conv4

def define_discriminator(image_shape):

    input_image = Input(shape=image_shape)  #Image we want to convert to another image - Fluorescnce image

    d = Conv2D(32, (3, 3), strides=2, padding='same')(input_image)

    #first block
    x1 = disc_block(d, 64)
    x1 = disc_block(x1, 64)
    #second block
    x2 = disc_block(x1, 128)
    x2 = disc_block(x2, 128)
    #third block
    x3 = disc_block(x2, 256)
    x3 = disc_block(x3, 256)
    #fourth block
    x4 = disc_block(x3, 512)
    x4 = disc_block(x4, 512)
    #fifth block
    x5 = disc_block(x4, 512)
    x5 = disc_block(x5, 512)

    #Pooling
    pooled = AveragePooling2D(pool_size=(4,4), strides=2, padding='same')(x5)

    #Fully-connected layers
    FC_layer = Flatten()(pooled)
    FC_layer = Dense(1024, activation='relu')(FC_layer)
    output = Dense(1, activation='sigmoid')(FC_layer)

    opt = Adam(learning_rate=1e-5)
    model = Model(input_image, output)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def conv_block(inputs, channel_size, blocks=2):

    conv = Conv2D(channel_size, kernel_size=3, padding='same')(inputs)
    conv = LeakyReLU(alpha=0.1)(conv)

    for i in range(blocks):
        conv = Conv2D(channel_size, kernel_size=3, padding='same')(conv)
        conv = LeakyReLU(alpha=0.1)(conv)

    return conv

#downsampling+residual
def downsampling_block(inputs, filters):
    first = conv_block(inputs, filters)
    residual = Conv2D(filters, (1, 1), padding='same')(inputs)
    x = add([first, residual])
    x = LeakyReLU(alpha=0.1)(x)
    pooled = AveragePooling2D((2, 2))(x)
    return pooled, first

#upsampling
def upsampling_block(inputs, skip_connection, filters):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = Concatenate()([x, skip_connection])
    x = conv_block(x, filters)

    return x


def define_generator(image_shape):
    inputs = Input(shape=image_shape)

    # Downsampling blocks
    downsampling1, skip1 = downsampling_block(inputs, 64)
    downsampling2, skip2 = downsampling_block(downsampling1, 128)
    downsampling3, skip3 = downsampling_block(downsampling2, 256)
    downsampling4, skip4 = downsampling_block(downsampling3, 512)

    # Upsampling blocks
    upsampling1 = upsampling_block(downsampling4, skip4, 256)
    upsampling2 = upsampling_block(upsampling1, skip3, 128)
    upsampling3 = upsampling_block(upsampling2, skip2, 64)
    upsampling4 = upsampling_block(upsampling3, skip1, 64)

    # Output
    outputs = Conv2D(3, (1, 1))(upsampling4)
    outputs = LeakyReLU(alpha=0.1)(outputs)

    opt = Adam(learning_rate=1e-5)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def ssim_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.uint8)
    y_pred = tf.cast(y_pred, tf.uint8)

    #SSIM
    ssim_value = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

    #the loss
    return 1 - ssim_value

def define_gan(gen_model, disc_model, image_shape, weights=None):

    for layer in disc_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable=False

    source_im = Input(shape=image_shape)

    generator_output = gen_model(source_im)

    discriminator_output = disc_model(generator_output)

    model = Model(source_im, [discriminator_output, generator_output])

    opt = Adam(learning_rate=1e-5)

    model.compile(loss=['binary_crossentropy', 'mse', ssim_loss], optimizer=opt, loss_weights=weights)

    return model

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

def generate_real_samples(data, n_samples, patch_shape):

    conf, fluo = data
    idx = randint(0, conf.shape[0], n_samples)
    conf_img, fluo_img = conf[idx], fluo[idx]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))

    return [conf_img, fluo_img], y

def generate_fake_samples(g_model, samples, patch_shape):

    predicted = g_model.predict(samples)
    y = np.zeros((len(predicted), patch_shape, patch_shape, 1))

    return predicted, y

def generate_validation_samples(data, n_samples):

    conf, fluo = data
    idx = randint(0, conf.shape[0], n_samples)
    conf_img, fluo_img = conf[idx], fluo[idx]

    return [conf_img, fluo_img]

def train_GAN(d_model, g_model, gan_model, data, save_path, epochs = 300, n_batch = 1, model_weights=None):

    n_patch = d_model.output_shape[1]

    conf_images, fluo_images = data

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

def data_loader(lr_path, hr_path, img_size):
    
    img_width = img_size
    img_height = img_size
    
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

lr_path = "trial_data/fluo_images_2D"
hr_path = "trial_data/conf_images_2D"

fluo_images, conf_images = data_loader(lr_path=lr_path, hr_path=hr_path, img_size=256)

image_shape = fluo_images.shape[1:]
d_model = define_discriminator(image_shape=image_shape)
g_model = define_generator(image_shape=image_shape)
loss_weights = [1, 0.01, 0.1]
gan_model = define_gan(g_model, d_model, image_shape=image_shape, weights=loss_weights)
save_path = '/Users/dominikpanek/Downloads/trial/'
model_weights = '/Users/dominikpanek/Downloads/UNET_GAN_model6_Adv1_MSE001_SSIM01_epoch_200006.h5'
data = [conf_images, fluo_images]

train_GAN(d_model=d_model, 
          g_model=g_model, 
          gan_model=gan_model, 
          save_path=save_path, 
          data=data, 
          epochs=1, 
          n_batch=1, 
          model_weights=model_weights)