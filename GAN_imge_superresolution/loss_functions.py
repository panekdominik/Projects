import tensorflow as tf

def ssim_loss(y_true, y_pred):
    
    y_true = tf.cast(y_true, tf.uint8)
    y_pred = tf.cast(y_pred, tf.uint8)

    #SSIM
    ssim_value = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

    #the loss
    return 1 - ssim_value