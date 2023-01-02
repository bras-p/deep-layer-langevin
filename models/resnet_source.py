import tensorflow as tf

# https://www.analyticsvidhya.com/blog/2021/08/how-to-code-your-resnet-from-scratch-in-tensorflow/#h2_5



def identity_block(x, filters, mode='resnet'):
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    if not mode=='vgg':
        x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolutional_block(x, filters, mode='resnet'):
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filters, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    if not mode=='vgg':
        # Processing Residue with conv(1,1)
        x_skip = tf.keras.layers.Conv2D(filters, (1,1), strides = (2,2))(x_skip)
        x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x


# def makeResNet(input_shape, classes, filter_size, block_layers, hidden_units = 64, zero_padding=(0,0)):
#     # Step 1 (Setup Input Layer)
#     x_input = tf.keras.layers.Input(input_shape)
#     x = tf.keras.layers.ZeroPadding2D(zero_padding)(x_input)
#     # Step 2 (Initial Conv layer along with maxPool)
#     x = tf.keras.layers.Conv2D(filter_size, kernel_size=7, strides=2, padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
#     # Step 3 Add the Resnet Blocks
#     for i in range(len(block_layers)):
#         if i == 0:
#             # For sub-block 1 Residual/Convolutional block not needed
#             for j in range(block_layers[i]):
#                 x = identity_block(x, filter_size)
#         else:
#             # One Residual/Convolutional Block followed by Identity blocks
#             # The filter size will go on increasing by a factor of 2
#             filter_size = filter_size*2
#             x = convolutional_block(x, filter_size)
#             for j in range(block_layers[i] - 1):
#                 x = identity_block(x, filter_size)
#     # Step 4 End Dense Network
#     x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(hidden_units, activation = 'relu')(x)
#     x = tf.keras.layers.Dense(classes)(x)
#     model = tf.keras.models.Model(inputs = x_input, outputs = x)
#     return model


def makeResNet(input_shape, classes, filter_size, block_layers, mode='resnet', zero_padding=(0,0)):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.ZeroPadding2D(zero_padding)(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(filter_size, kernel_size=3, padding='same')(x)
    # kernel_size is 3 for CIFAR-10, 7 in general
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x) # not this layer for CIFAR-10
    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size, mode)
        else:
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size, mode)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size, mode)
    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(hidden_units, activation = 'relu')(x)
    # no hidden layer in original architecture
    x = tf.keras.layers.Dense(classes)(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    return model
