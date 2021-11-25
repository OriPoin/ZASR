import tensorflow as tf
from tensorflow.keras import Model, layers

class MiniRnnt(Model):
    # Set layers.
    def __init__(self):
        super(MiniRnnt, self).__init__()
        # RNN-T (LSTM) hidden layer.
        self.conv1 = layers.Conv2D(32, kernel_size=3)
        


    # Set forward pass.
    def call(self, x, is_training=False):
        # LSTM layer.
        x = self.lstm_layer(x)
        # Output layer (num_classes).
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# Build LSTM model.
lstm_net = MiniRnnt()