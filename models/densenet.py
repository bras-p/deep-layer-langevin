import tensorflow as tf
from .base import ModelBuilder


class DenseNet(ModelBuilder):
    def __init__(
        self,
        input_shape = (32,32,3),
        growth_rate = 12,
        block_layers = [12,12,12],
        initial_conv_channels = 16,
        classes = 10,
        zero_padding = (0,0),
    ):
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.block_layers = block_layers
        self.initial_conv_channels = initial_conv_channels
        self.classes = classes
        self.zero_padding = zero_padding

    def densenet_layer(self, x):
        y = tf.keras.layers.BatchNormalization(axis=3)(x)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Conv2D(self.growth_rate, (3,3), padding="same")(y)
        x = tf.concat([x,y], axis=3)
        return x

    def transition_block(self, x):
        nb_channels = x.shape[-1]
        y = tf.keras.layers.BatchNormalization(axis=3)(x)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.Conv2D(nb_channels, (1,1), padding="same")(y)
        y = tf.keras.layers.Activation('relu')(y)
        y = tf.keras.layers.AveragePooling2D((2,2), strides=(2,2), padding="same")(y)
        return y

    def getModel(self):
        x_input = tf.keras.layers.Input(self.input_shape)
        x = tf.keras.layers.ZeroPadding2D(self.zero_padding)(x_input)
        x = tf.keras.layers.Conv2D(self.initial_conv_channels, (3,3), padding="same")(x)

        for i,j in enumerate(self.block_layers):
            for _ in range(j):
                x = self.densenet_layer(x)
            if i != len(self.block_layers)-1:
                x = self.transition_block(x)
        
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense((self.classes))(x)

        model = tf.keras.models.Model(inputs = x_input, outputs = x)
        return model