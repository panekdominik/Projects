from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Input
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

def model(image_width, image_height, n_channels, filters=32, kernel=(3, 3), blocks=2, lr=1e-5):
    
    # CNN model
    input_layer = Input(shape=(image_width, image_height, n_channels))
    conv1 = Conv2D(filters, kernel, activation='relu')(input_layer)
    pool1 = MaxPooling2D(kernel)(conv1)

    conv = conv_block(input=pool1, filters=filters, blocks=blocks)

    FC_layer = Flatten()(conv)
    FC_layer = Dense(64, activation='relu')(FC_layer)
    FC_layer = Dropout(0.5)(FC_layer)
    output = Dense(5, activation='softmax')(FC_layer)

    model = Model(input_layer, output)
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