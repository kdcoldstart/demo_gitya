'''This is the model for taxi drivers dataset'''

import tensorflow as tf
from tensorflow.keras import layers

class Model(tf.keras.Model):
    def __init__(self, config):
        """
        Initialize the model with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the model.
        """
        super(Model, self).__init__()
        self.hidden_size = 128
        self.lstm = layers.LSTM(self.hidden_size, dropout=0.1, return_sequences=True)
        self.lstm2 = layers.LSTM(self.hidden_size // 4, dropout=0.1)
        self.dense = layers.Dense(config.get('forecast_horizon'))

    def call(self, inputs):
        """
        Forward pass for the model.
        
        Args:
            inputs (Tensor): Input tensor for the model.
            
        Returns:
            Tensor: Output tensor after passing through the model.
        """
        x = self.lstm(inputs)
        x = layers.Dropout(0.5)(x)
        x = self.lstm2(x)
        x = layers.Dropout(0.5)(x)
        return self.dense(x)

    def loss(self, pred, truth):
        """
        Compute the loss between the predicted and true values.
        
        Args:
            pred (Tensor): Predicted values.
            truth (Tensor): True values.
            
        Returns:
            Tensor: Computed loss.
        """
        return tf.reduce_mean(tf.square(pred - truth))