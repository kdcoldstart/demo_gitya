'''This is the model for taxi drivers dataset'''

import tensorflow as tf
from tensorflow.keras import layers

class Model(tf.keras.Model):
    def __init__(self, config):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.lstm = layers.LSTM(self.hidden_size, dropout=0.1, return_sequences=True)
        self.lstm2 = layers.LSTM(self.hidden_size // 4, dropout=0.1)
        self.dense = layers.Dense(config.get('forecast_horizon'))

    def call(self, inputs):
        x = self.lstm(inputs)
        x = layers.Dropout(0.5)(x)
        x = self.lstm2(x)
        x = layers.Dropout(0.5)(x)
        return self.dense(x)

    def loss(self, pred, truth):
        return tf.reduce_mean(tf.square(pred - truth))