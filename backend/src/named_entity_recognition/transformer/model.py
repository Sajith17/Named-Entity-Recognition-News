import tensorflow as tf
from named_entity_recognition.transformer.encoder import *

class NERModel(tf.keras.Model):

  def __init__(self, num_tags, num_layers, embedding_dim, num_heads, fully_connected_dim,
               vocab_size, max_positional_encoding,dropout_rate=0.1,
               layernorm_eps=1e-6):

    super(NERModel, self).__init__()

    self.encoder = Encoder(     num_layers=num_layers,
                                embedding_dim=embedding_dim,
                                num_heads=num_heads,
                                fully_connected_dim=fully_connected_dim,
                                vocab_size=vocab_size,
                                maximum_position_encoding=max_positional_encoding,
                                dropout_rate=dropout_rate,
                                layernorm_eps=layernorm_eps)
    self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)
    self.ffn = tf.keras.layers.Dense(fully_connected_dim, activation='relu')
    self.ffn_final = tf.keras.layers.Dense(num_tags,activation='softmax')

  def call(self, x, training):
    x, mask = x
    mask = mask[:,tf.newaxis,:]
    x = self.encoder(x,training, mask)
    x = self.dropout1(x, training = training)
    x = self.ffn(x)
    x = self.dropout2(x, training=training)
    x = self.ffn_final(x)
    return x
