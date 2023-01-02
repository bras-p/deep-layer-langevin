import tensorflow as tf
from .base import ModelBuilder
from .resnet_source import makeResNet


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

    def getModel(self):
        model = makeResNet(
            self.input_shape,
            self.classes,
            self.filters,
            self.block_layers,
            self.mode,
            self.zero_padding
        )
        return model


