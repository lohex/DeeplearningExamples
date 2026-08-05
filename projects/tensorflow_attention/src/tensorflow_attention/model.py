"""Baseline and temporal-attention classifiers."""

import tensorflow as tf
from tensorflow import keras


def build_baseline_model(
    input_length: int = 288,
    num_classes: int = 3,
    filters: int = 16,
) -> keras.Model:
    """Build a two-branch convolutional baseline without attention."""
    inputs = keras.Input((input_length, 1), name="trajectory")
    level_features = _encoder(inputs, filters=filters, prefix="level")
    derivative = keras.layers.Lambda(
        lambda values: values[:, 1:] - values[:, :-1], name="first_difference"
    )(inputs)
    derivative_features = _encoder(derivative, filters=filters, prefix="difference")
    merged = keras.layers.Concatenate()([level_features, derivative_features])
    pooled = keras.layers.GlobalAveragePooling1D()(merged)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(pooled)
    return _compile(keras.Model(inputs, outputs, name="convolutional_baseline"))


def build_attention_model(
    input_length: int = 288,
    num_classes: int = 3,
    filters: int = 16,
) -> keras.Model:
    """Build a cross-attention classifier over levels and differences."""
    inputs = keras.Input((input_length, 1), name="trajectory")
    level_features = _encoder(inputs, filters=filters, prefix="level")
    derivative = keras.layers.Lambda(
        lambda values: values[:, 1:] - values[:, :-1], name="first_difference"
    )(inputs)
    derivative_features = _encoder(derivative, filters=filters, prefix="difference")
    attended = keras.layers.Attention(name="temporal_attention")(
        [level_features, derivative_features]
    )
    pooled = keras.layers.GlobalAveragePooling1D()(attended)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(pooled)
    return _compile(keras.Model(inputs, outputs, name="temporal_attention_classifier"))


def build_attention_probe(model: keras.Model) -> keras.Model:
    """Return a model that exposes temporal attention scores."""
    layer = model.get_layer("temporal_attention")
    return keras.Model(model.inputs, layer.output, name="attention_probe")


def _encoder(inputs: tf.Tensor, *, filters: int, prefix: str) -> tf.Tensor:
    values = keras.layers.Conv1D(filters, 12, strides=3, padding="same", activation="relu", name=f"{prefix}_conv_1")(inputs)
    values = keras.layers.BatchNormalization(name=f"{prefix}_batch_norm_1")(values)
    values = keras.layers.Conv1D(filters * 2, 7, strides=2, padding="same", activation="relu", name=f"{prefix}_conv_2")(values)
    return keras.layers.BatchNormalization(name=f"{prefix}_batch_norm_2")(values)


def _compile(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
    )
    return model
