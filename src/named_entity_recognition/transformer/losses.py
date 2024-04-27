import tensorflow as tf


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ner_loss", reduction='none'):
        super().__init__(name=name, reduction=reduction)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def call(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, -1))
        mask = tf.cast(mask, dtype=tf.int32)

        loss = self.loss_object(y_true*mask, y_pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return tf.reduce_sum(loss)/tf.reduce_sum(mask)
    