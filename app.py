import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageEnhance
import matplotlib.cm as cm
import requests
from io import BytesIO

# Load model
model = MobileNetV2(weights="imagenet")

st.title("Image Classification Application")
st.write("Upload an image and let the model classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_resized = resize(image, (224, 224), anti_aliasing=True)
    image_resized = (image_resized * 255).astype(np.uint8)
    #or
    #image_resized = tf.image.resize(image, (224, 224)).numpy()

    img_array = np.array(image_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array_expanded)

    # Predict
    predictions = model.predict(processed_img)
    decoded_preds = decode_predictions(predictions, top=3)[0]

    st.write("### Predictions:")
    for i, (imagenet_id, label, prob) in enumerate(decoded_preds):
        st.write(f"**{i+1}. {label}** ({prob*100:.2f}%)")
        st.progress(int(prob * 100))

    # -------- Grad-CAM --------
    st.write("### Grad-CAM Visualization")

    # Define a model that maps input image to activations & predictions
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("Conv_1").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_img)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the output neuron (target class) with respect to feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Mean intensity of the gradients for each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by its corresponding gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Resize heatmap to original image size
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)

    # Apply colormap (use matplotlib colormap)
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(np.array(heatmap_resized) / 255.0)
    colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    colored_heatmap_img = Image.fromarray(colored_heatmap)

    # Blend original image and heatmap
    blended = Image.blend(image, colored_heatmap_img, alpha=0.4)

    # Show result
    st.image(blended, caption="Grad-CAM Heatmap (No OpenCV)", use_column_width=True)
