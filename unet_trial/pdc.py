import tensorflow as tf

def PDC(input_layer):
    # Convolutional Layer 1
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), dilation_rate=1, activation='relu', padding='same')(input_layer)
    
    # Convolutional Layer 2
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), dilation_rate=2, activation='relu', padding='same')(input_layer)
    
    # Convolutional Layer 3
    conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), dilation_rate=4, activation='relu', padding='same')(input_layer)
    
    # Convolutional Layer 4
    conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), dilation_rate=8, activation='relu', padding='same')(input_layer)
    
    # Concatenate the outputs of all convolutional layers
    concatenated_layers = tf.keras.layers.concatenate([conv1, conv2, conv3, conv4], axis=-1)
    output_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(concatenated_layers)
    output_layer = tf.keras.layers.BatchNormalization()(output_layer)
    return output_layer

# Define input shape
input_shape = (64, 64, 3)  # Example input shape

# Create input layer
input_layer = tf.keras.layers.Input(shape=input_shape)

# Apply parallel convolutional layers
output_layer = PDC(input_layer)

# Create model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Print model summary
model.summary()
