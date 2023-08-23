from numpy.random import randint
import numpy as np

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