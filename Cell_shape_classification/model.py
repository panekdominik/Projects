from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, SimpleRNN,
                                     BatchNormalization, Dropout, Input, LSTM, GRU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

def conv_block(input, filters, blocks, kernel=(3,3)):

    for i in range(blocks):
        conv = Conv2D(filters*2*(i+1), kernel, activation='relu')(input)
        conv = BatchNormalization()(conv)
        conv = MaxPooling2D(kernel)(conv)
        
        if i % 2 == 0:
            conv = Dropout(0.25)(conv)

    return conv

def convolutional_model(image_width, image_height, n_channels, filters=32, kernel=(3, 3), blocks=2, lr=1e-5, num_classes=3):
    
    # CNN model
    input_layer = Input(shape=(image_width, image_height, n_channels))
    conv1 = Conv2D(filters, kernel, activation='relu')(input_layer)
    pool1 = MaxPooling2D(kernel)(conv1)

    conv = conv_block(input=pool1, filters=filters, blocks=blocks)

    FC_layer = Flatten()(conv)
    FC_layer = Dense(filters*2, activation='relu')(FC_layer)
    FC_layer = Dropout(0.5)(FC_layer)
    output = Dense(num_classes, activation='softmax')(FC_layer)

    model = Model(input_layer, output)
    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def recurrent_model(seq_length, image_width, image_height, n_channels, num_classes=3, units=16, rnn_type='lstm', include_conv = False, lr = 1e-5):

    inputs = Input(shape=(seq_length, image_width, image_height, n_channels))
    if include_conv == True:
        x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)
        x = TimeDistributed(Dropout(rate=0.25))(x)
        x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Flatten())(inputs)
    if rnn_type == 'lstm':
        x = LSTM(units, return_sequences=True)(x)
    elif rnn_type == 'gru':
        x = GRU(units, return_sequences=True)(x)
    elif rnn_type == 'simple_rnn':
        x = SimpleRNN(units, return_sequences=True)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x)
    opt = Adam(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model 

def callbacks(model_save, performance_save, patience=20):

    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    model_checkpoint = ModelCheckpoint(model_save,
                                    verbose=1, save_best_only=True,
                                    save_weights_only=False)
    csv_logger = CSVLogger(performance_save,
                        separator=',', append=False)
    callbacks = [lr_scheduler, early_stopping, model_checkpoint, csv_logger]

    return callbacks