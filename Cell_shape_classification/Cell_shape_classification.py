import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import glob
import pandas as pd
import imageio as io
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, GRU, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc
from sklearn.utils import class_weight


main_directory_11 = 'D:/Tomek/Seria 11/Maski_seria11'
main_directory_13 = 'D:/Tomek/Seria 13/Maski_seria13'
main_directory_15 = 'D:/Tomek/Seria 15/Maski_seria15'
main_directory_17 = 'D:/Tomek/Seria 17/Maski_seria17'
main_directory_21 = 'D:/Tomek/Seria 21/Maski_seria21'

folder_list_11 = os.listdir(main_directory_11)
folder_list_13 = os.listdir(main_directory_13)
folder_list_15 = os.listdir(main_directory_15)
folder_list_17 = os.listdir(main_directory_17)
folder_list_21 = os.listdir(main_directory_21)


seria_11 = []
for i, j in enumerate(folder_list_11):
    seria_ = sorted(glob.glob("D:Tomek/Seria 11/Maski_seria11/{}/*".format(j)))
    #print(seria_)
    seria_11.append(seria_)
seria_11 = np.array(seria_11)

seria_13 = []
for i, j in enumerate(folder_list_13):
    seria_ = sorted(glob.glob("D:Tomek/Seria 13/Maski_seria13/{}/*".format(j)))
    #print(seria_)
    seria_13.append(seria_)
seria_13 = np.array(seria_13)

seria_15 = []
for i, j in enumerate(folder_list_15):
    seria_ = sorted(glob.glob("D:Tomek/Seria 15/Maski_seria15/{}/*".format(j)))
    #print(seria_)
    seria_15.append(seria_)
seria_15 = np.array(seria_15)

seria_17 = []
for i, j in enumerate(folder_list_17):
    seria_ = sorted(glob.glob("D:Tomek/Seria 17/Maski_seria17/{}/*".format(j)))
    #print(seria_)
    seria_17.append(seria_)
seria_17 = np.array(seria_17)

seria_21 = []
for i, j in enumerate(folder_list_21):
    seria_ = sorted(glob.glob("D:Tomek/Seria 21/Maski_seria21/{}/*".format(j)))
    #print(seria_)
    seria_21.append(seria_)
seria_21 = np.array(seria_21)

x = np.append(seria_11, seria_13)
y = np.append(x, seria_15)
z = np.append(y, seria_17)
data = np.append(z, seria_21)

# single image dimensions
image_height = 32
image_width = 32
n_channels = 1


image_data = []
for chunk in data:
    img = io.imread(chunk)
    img = np.invert(img)
    img = img/255
    imgage = resize(img, (image_width, image_height, n_channels), preserve_range = True)
    image_data.append(imgage)


image_data = np.array(image_data)
image_data = image_data.reshape((48, 161, image_width, image_height, n_channels))

labels = pd.read_table('D:/Tomek/labels.txt', header=None)
labels = np.array(labels[0])

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

num_classes = len(label_encoder.classes_)
labels = to_categorical(labels_encoded, num_classes=num_classes)
labels = labels.reshape((48, 161, 5))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# CNN-RNN model
model = Sequential()
model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'), input_shape=(161, image_width, image_height, n_channels)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

# model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu')))
# model.add(TimeDistributed(BatchNormalization()))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

# model.add(TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu')))
# model.add(TimeDistributed(BatchNormalization()))
# model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

model.add(TimeDistributed(Flatten()))
model.add(GRU(units=128, return_sequences=True))
model.add(TimeDistributed(Dense(units=num_classes, activation='softmax')))

model.summary()
# Compile the model
opt = Adam(learning_rate=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# callbacks
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
callbacks = [lr_scheduler]

n_epochs = 300

# train the model
history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=16, validation_data=(X_test, y_test), callbacks=callbacks)
model.save('cell_shape_model_1.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')

epochs = np.arange(1, n_epochs+1, 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

fig, ax = plt.subplots(1, 2, figsize=(13,6))
ax[0].plot(epochs, train_loss, label = 'Training loss')
ax[0].plot(epochs, val_loss, label = 'Validation loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].plot(epochs, train_acc, label = 'Training accuracy')
ax[1].plot(epochs, val_acc, label = 'Validation accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

preds = model.predict(X_test)
predicted_classes = np.argmax(preds, axis=1)
correct_predictions = np.equal(predicted_classes, np.argmax(y_test, axis=1))

fpr, tpr, thresholds = roc_curve(y_test.ravel(), preds.ravel())
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')