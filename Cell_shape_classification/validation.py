import numpy as np
import matplotlib.pyplot as plt

def model_validation(labels_path, data, model, data_files):

    labels = np.loadtxt(labels_path)
    labels_list = []
    for i in labels:
        i  = int(i)
        labels_list.append(i)

    incorrect_preds = []
    for id, img in enumerate(data):
        image = np.expand_dims(img, axis=0)
        prediction = model.predict(image, verbose=0)
        label = np.argmax(prediction, axis=1)+1
        if label != labels[id]:
            incorrect_preds.append(id)
    incorrect_preds=np.array(incorrect_preds)
    print('Number of incorrect predictions:', len(incorrect_preds))

    id = np.random.choice(incorrect_preds)
    x = model.predict(np.expand_dims(data[id], axis=0), verbose=0)
    print('Cell ID:', id,
        '\n', 'Prediction array:', x[0],
        '\n', 'Predicted class:', np.argmax(x, axis=1)[0]+1,
        '\n', 'Real class:', labels[id],
        '\n', 'Image path:', data_files[id])
    plt.imshow(data[id], cmap='gray')