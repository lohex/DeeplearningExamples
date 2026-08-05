"""One-dimensional convolutional classifier used by Grad-CAM."""

from tensorflow import keras


def build_model(
    input_length: int = 288,
    num_classes: int = 3,
    filters: int = 32,
) -> keras.Model:
    inputs = keras.Input((input_length, 1), name="trajectory")
    values = keras.layers.Conv1D(filters, 12, strides=3, padding="same", activation="relu", name="feature_conv_1")(inputs)
    values = keras.layers.BatchNormalization()(values)
    values = keras.layers.Conv1D(filters // 2, 7, strides=2, padding="same", activation="relu", name="gradcam_features")(values)
    values = keras.layers.BatchNormalization()(values)
    values = keras.layers.GlobalAveragePooling1D()(values)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(values)
    model = keras.Model(inputs, outputs, name="timecourse_gradcam_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )
    return model
