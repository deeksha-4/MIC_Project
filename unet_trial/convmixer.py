import tensorflow as tf
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))

def ConvMixer(input_layer):
    depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(input_layer)
    # GELU Activation
    gelu_activation = tf.keras.layers.Lambda(lambda x: gelu(x))(depthwise_conv)
    z_ = tf.keras.layers.BatchNormalization()(gelu_activation)
    z_input = tf.keras.layers.Add()([z_, input_layer])
    pointwise_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation=None, padding='same')(z_input)
    gelu_activation_2 = tf.keras.layers.Lambda(lambda x: gelu(x))(pointwise_conv)

    output_layer = tf.keras.layers.BatchNormalization()(gelu_activation_2)
    return output_layer

# Define input shape
input_shape = (64, 64, 3)  # Example input shape

# Create input layer
input_layer = tf.keras.layers.Input(shape=input_shape)

# Apply parallel convolutional layers
output_layer = ConvMixer(input_layer)

# Create model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Print model summary
model.summary()
