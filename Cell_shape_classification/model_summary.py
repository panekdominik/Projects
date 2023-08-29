from sklearn.metrics import roc_curve, auc
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def model_performnace(path, save=False, save_path=None):
    
    data = pd.read_csv(path, sep=',')
    fig, ax = plt.subplots(1, 2, figsize=(13,5))
    sns.lineplot(x=data['epoch'], y=data['accuracy'], ax=ax[0], label='Training')
    sns.lineplot(x=data['epoch'], y=data['val_accuracy'], ax=ax[0], label='Testing')
    sns.lineplot(x=data['epoch'], y=data['loss'], ax=ax[1], label='Training')
    sns.lineplot(x=data['epoch'], y=data['val_loss'], ax=ax[1], label='Testing')
    ax[0].set_xlabel('Epoch')
    ax[1].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_ylabel('Loss')
    plt.tight_layout()
    plt.show()

    if save==True:
        plt.savefig(save_path, dpi=600, transparent=True)


def model_evaluation(model, data, labels, n_classes, save=False, save_path=None):
    
    preds = model.predict(data)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.5f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    if save==True:
        plt.savefig(save_path, dpi=600, transparent=True)

def model_metrics(model, data, y_test, names = None, save=False, save_path=None):
    
    preds = model.predict(data)
    y_pred = np.argmax(preds, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    print(classification_report(y_true, y_pred, target_names=names))
    heat_map = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=names)
    names_dict = {}
    for index, element in enumerate(names):
        names_dict[index] = element
    heat_map.rename(index=names_dict, inplace=True)

    sns.heatmap(heat_map, annot=True, cmap='viridis')
    plt.tight_layout()
    plt.xlabel('Predicted Classes')
    plt.ylabel('True Classes')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    if save == True:
        plt.savefig(save_path, dpi=600, transparent=True)
