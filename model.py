import tensorflow as tf


def create_model(num_frozen_layers, input_shape, num_class):
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Add new output layers
    x = tf.keras.layers.GlobalMaxPooling2D()(base_model.layers[-1].output)
    outputs = tf.keras.layers.Dense(num_class, activation=tf.keras.activations.softmax)(x)

    tl_model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs)

    # Freeze first 150 layers
    for i in range(0, num_frozen_layers):
        tl_model.layers[i].trainable = False

    return tl_model
    # tl_model.summary(line_length=  200)
