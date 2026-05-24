import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import cv2

def find_last_conv_layer(model):
    """Dynamically finds the last convolutional layer in a nested Keras model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Generates a Grad-CAM heatmap for a given image and model."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            return np.zeros((224, 224))

    try:
        grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    except ValueError:
        # Handle nested models (like Functional layers containing the base architecture)
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                try:
                    conv_output = layer.get_layer(last_conv_layer_name).output
                    grad_model = Model(model.inputs, [conv_output, model.output])
                    break
                except ValueError:
                    continue
        else:
            return np.zeros((224, 224))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        
        # Handle cases where model outputs are wrapped in lists
        if isinstance(predictions, list):
            predictions = predictions[0]
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]
            
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return np.zeros((224, 224))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """Overlays the Grad-CAM heatmap onto the original image."""
    img_array = img_array.astype('uint8')
    heatmap_resized = cv2.resize(np.uint8(255 * heatmap), (img_array.shape[1], img_array.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
