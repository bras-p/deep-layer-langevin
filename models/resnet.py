import tensorflow as tf
from .base import ModelBuilder


class ResNet(ModelBuilder): # can also instantiate the VGG model
    def __init__(
        self,
        input_shape = (32,32,3),
        filters = 32,
        block_layers = [4,4,4],
        hidden_units = 512,
        classes = 10,
        zero_padding = (0,0),
        mode = 'resnet' # equals to 'resnet' or to 'vgg'
    ):
        self.input_shape = input_shape
        self.filters = filters
        self.block_layers = block_layers
        self.hidden_units = hidden_units
        self.classes = classes
        self.zero_padding = zero_padding
        self.mode = mode

    def identity_block(self, x, filter_size):
        x_skip = x
        x = tf.keras.layers.Conv2D(filter_size, (3,3), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filter_size, (3,3), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        if not self.mode=='vgg':
            x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def convolutional_block(self, x, filter_size):
        x_skip = x
        x = tf.keras.layers.Conv2D(filter_size, (3,3), padding = 'same', strides = (2,2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filter_size, (3,3), padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        if not self.mode=='vgg':
            x_skip = tf.keras.layers.Conv2D(filter_size, (1,1), strides = (2,2))(x_skip)
            x = tf.keras.layers.Add()([x, x_skip])     
        x = tf.keras.layers.Activation('relu')(x)
        return x


    def getModel(self):
        filter_size = self.filters
        x_input = tf.keras.layers.Input(self.input_shape)
        x = tf.keras.layers.ZeroPadding2D(self.zero_padding)(x_input)
        x = tf.keras.layers.Conv2D(filter_size, kernel_size=3, padding='same')(x) # kernel_size is 3 for CIFAR-10, 7 in general
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        for i in range(len(self.block_layers)):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(self.block_layers[i]):
                    x = self.identity_block(x, filter_size)
            else:
                filter_size = filter_size*2
                x = self.convolutional_block(x, filter_size)
                for j in range(self.block_layers[i] - 1):
                    x = self.identity_block(x, filter_size)

        x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.classes)(x)

        model = tf.keras.models.Model(inputs = x_input, outputs = x)
        return model


