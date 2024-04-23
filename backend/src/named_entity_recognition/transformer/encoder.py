import tensorflow as tf
import numpy as np

def positional_encoding(positions, d_model):

    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def FullyConnected(embedding_dim, fully_connected_dim):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(fully_connected_dim, activation='relu'),
      tf.keras.layers.Dense(embedding_dim)
  ])

class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self, embedding_dim, num_heads, fully_connected_dim,
               dropout_rate=0.1, layernorm_eps=1e-6):

    super(EncoderLayer, self).__init__()

    self.mha = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
        dropout=dropout_rate
    )

    self.ffn = FullyConnected(
        embedding_dim=embedding_dim,
        fully_connected_dim=fully_connected_dim
    )

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

    self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, training, mask):

    self_mha_output = self.mha(x,x,x,mask)
    skip_x_attention = self.layernorm1(x+self_mha_output)
    ffn_output = self.ffn(skip_x_attention)
    ffn_output = self.dropout_ffn(ffn_output, training=training)
    encoder_layer_out = self.layernorm2(skip_x_attention+ffn_output)

    return encoder_layer_out
  
class Encoder(tf.keras.layers.Layer):

  def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, vocab_size,
               maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
    super(Encoder, self).__init__()

    self.embedding_dim = embedding_dim

    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.embedding_dim)

    self.enc_layers = [EncoderLayer(embedding_dim=embedding_dim,
                                    num_heads=num_heads,
                                    fully_connected_dim=fully_connected_dim,
                                    dropout_rate=dropout_rate,
                                    layernorm_eps=layernorm_eps
                                    )
                      for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, training, mask):

     seq_len = tf.shape(x)[1]

     x = self.embedding(x)

     x*=tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))

     x+=self.pos_encoding[:,:seq_len,:]

     x = self.dropout(x, training=training)

     for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

     return x

  
