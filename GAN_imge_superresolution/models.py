from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, AveragePooling2D, LeakyReLU, Concatenate
from tensorflow.keras.layers import LeakyReLU, Dense, Input, add, Flatten, BatchNormalization
from loss_functions import ssim_loss


### Building a discriminator form convolutional blocks

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

### Building a generator form convoluional, downsampling and upsampling blocks

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

### GAN mdoel
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