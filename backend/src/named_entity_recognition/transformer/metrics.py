import tensorflow as tf


def masked_acc(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, -1))
    mask = tf.cast(mask,dtype=tf.float32)
    pred = tf.cast(tf.argmax(y_pred,axis=-1),dtype=tf.float32)
    correct = tf.cast(tf.equal(y_true,pred),dtype=tf.float32)
    correct*=mask
    return tf.reduce_sum(correct)/tf.reduce_sum(mask)


