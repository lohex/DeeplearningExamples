"""Functional Grad-CAM utilities for one-dimensional convolutional models."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from tensorflow import keras


def compute_gradcam(
    model: keras.Model,
    trajectory: NDArray[np.floating],
    *,
    layer_name: str = "gradcam_features",
    class_index: int | None = None,
) -> NDArray[np.floating]:
    """Return a normalized heatmap for one trajectory and target class."""
    batch = np.asarray(trajectory, dtype=np.float32)
    if batch.ndim == 1:
        batch = batch[:, None]
    if batch.ndim == 2:
        batch = batch[None, ...]
    probe = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        features, predictions = probe(batch, training=False)
        target = tf.argmax(predictions[0]) if class_index is None else class_index
        score = predictions[:, target]
    gradients = tape.gradient(score, features)
    weights = tf.reduce_mean(gradients, axis=1, keepdims=True)
    heatmap = tf.reduce_sum(features * weights, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return heatmap.numpy()


def upsample_heatmap(heatmap: NDArray[np.floating], output_length: int) -> NDArray[np.floating]:
    """Interpolate a feature-map heatmap onto the input time axis."""
    source = np.linspace(0.0, 1.0, len(heatmap))
    target = np.linspace(0.0, 1.0, output_length)
    return np.interp(target, source, heatmap)


def plot_gradcam(
    axis: Axes,
    times: NDArray[np.floating],
    trajectory: NDArray[np.floating],
    heatmap: NDArray[np.floating],
) -> Axes:
    """Overlay temporal Grad-CAM relevance as a colored line."""
    signal = np.asarray(trajectory).squeeze()
    relevance = upsample_heatmap(heatmap, len(signal))
    points = np.column_stack((times, signal)).reshape(-1, 1, 2)
    segments = np.concatenate((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments, cmap="magma", norm=plt.Normalize(0.0, 1.0)
    )
    collection.set_array(relevance[:-1])
    collection.set_linewidth(2.5)
    axis.add_collection(collection)
    axis.autoscale()
    axis.set_xlabel("time")
    axis.set_ylabel("signal")
    return axis
