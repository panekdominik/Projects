from helpers import read_data_dir, img_reader, labels_reader
from sklearn.model_selection import train_test_split
from model import callbacks, convolutional_model, recurrent_model
from model_summary import model_performnace, model_evaluation, model_metrics
from validation import model_validation
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(prog="Cell shape classification script",
                                 description="This script allows to train convolutional nural network for cell shape classification",
                                 epilog="To learn more go to: https://github.com/panekdominik/PhD_projects/tree/main/Cell_shape_classification or read README file.")

parser.add_argument("--image_width", "-W", type=int, help="Width of the image.")
parser.add_argument("--image_height", "-H", type=int, help="Height of the image.")
parser.add_argument("--channels", "-C", type=int, default=1, help="Number of channels in an image. Default 1.")
parser.add_argument("--labels_path", "-LP", type=str, help="Path to the file containing labels.")
parser.add_argument("--model", "-M", type=str, default="recurrent", help="Chose the type of model you want to use. You can choose either convolutional or recurrent.")
parser.add_argument("--epochs", "-E", type=int, default=250, help="Number of iterations the model will run for. Default 250.")
parser.add_argument("--batch_size", "-BS", type=int, default=8, help="Number of images in one batch. Default 8.")
parser.add_argument("--learning_rate", "-LR", type=float, default=1e-5, help="Starting learning rate of learning. Default 1e-5.")
parser.add_argument("--kernel_size", "-KS", type=int, default=3, help="Size of the kernel in the convolutional block. Default value is 3 (matrix size 3x3).")
parser.add_argument("--blocks", "-B", type=int, default=2, help="Number of convolutional blocks in the model. Default value is 2.")
parser.add_argument("--model_save_path", "-MS", type=str, help="Path where the model will be saved.")
parser.add_argument("--logger_save_path", "-LS", type=str, help="Path where model performance will be saved.")
parser.add_argument("--cell_names", "-CN", type=str, default='Mesenchymal,Polygonal,Pseudopodial,Blebbing,Other', help="Name of the samples (given as delimited list). Default: Mesenchymal,Polygonal,Pseudopodial,Blebbing,Other")
parser.add_argument("--validation", "-V", type=bool, default=False, help="Default False. If true then ids of missclassified cells are returned")


args = parser.parse_args()
image_width = args.image_width
image_height = args.image_height
n_channels = args.channels
labels_path = args.labels_path
model_type = args.model
n_epochs = args.epochs
batch_size = args.batch_size
lr = args.learning_rate
kernel_size = args.kernel_size
blocks = args.blocks
save_path = args.model_save_path
log_path = args.logger_save_path
names = args.cell_names
validation = args.validation

# Set the path to the main directory containing the folders
dirs = ['/Users/dominikpanek/Downloads/dataset/Seria 11/Komórki_seria11', 
        '/Users/dominikpanek/Downloads/dataset/Seria 13/Komórki_seria13', 
        '/Users/dominikpanek/Downloads/dataset/Seria 15/Komórki_seria15', 
        '/Users/dominikpanek/Downloads/dataset/Seria 17/Komórki_seria17', 
        '/Users/dominikpanek/Downloads/dataset/Seria 21/Komórki_seria21']

data_files = read_data_dir(dirs)
image_data = img_reader(data_files=data_files, image_width=image_width, image_height=image_height, n_channels=n_channels)
labels, n_classes = labels_reader(path=labels_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.3, random_state=42)
print('Training shape:', X_train.shape, '\n','Testing shape:', X_test.shape)

# defining model and callbacks
if model_type == 'recurrent':
      model = convolutional_model(image_width=image_width, image_height=image_height, n_channels=n_channels, kernel=(kernel_size, kernel_size), blocks=blocks, lr=lr)
else:
     model = recurrent_model(image_width=image_width, image_height=image_height, n_channels=n_channels, lr=lr)

callback = callbacks(model_save=save_path, performance_save=log_path)

# train the model
start = datetime.now()
model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=callback)
stop = datetime.now()
ex_time = stop-start
print("Execution time is: ", ex_time)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Model Evaluations
model_performnace(path=log_path)
model_evaluation(model=model, data=X_test, labels=y_test, n_classes=n_classes)

names = [item for item in args.cell_names.split(',')]
model_metrics(model=model, data=X_test, y_test=y_test, names=names)

if validation == True:
    model_validation(labels_path=labels_path, data=image_data, model=model, data_files=data_files)