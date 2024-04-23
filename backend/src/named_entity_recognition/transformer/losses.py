import tensorflow as tf

class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

    def call(self, y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        loss_ = self.loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)