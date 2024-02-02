import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved model
loaded_model = load_model("Model/facial_expression_model.h5")

# Function to preprocess images
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Streamlit app
def main():
    st.title("Emotion Recognition App")

    # Upload a custom test image
    uploaded_file = st.file_uploader("Choose a custom test image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image using PIL
        try:
            img_pil = Image.open(uploaded_file)
            st.image(np.array(img_pil), caption="Original Image", use_column_width=True)

            # Convert PIL image to OpenCV format
            img_array = np.array(img_pil)  # Convert to NumPy array first
            custom_test_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Preprocess the custom test image
            custom_test_image_processed = preprocess_image(custom_test_image)

            # Make predictions on the custom test image
            prediction = loaded_model.predict(custom_test_image_processed)
            prediction_prob = prediction[0]

            # Map the predicted label to emotion class
            emotion_label = np.argmax(prediction[0])
            emotion_classes = {0: 'Happy', 1: 'Sad', 2: 'Angry'}
            predicted_emotion = emotion_classes[emotion_label]

            # Display the predicted emotion and confidence scores
            st.subheader("Prediction Results:")
            st.write(f"Predicted Emotion: {predicted_emotion}")
            st.write(f"Confidence [Happy, Sad, Angry]: {prediction_prob}")

            # Display the processed custom test image using matplotlib
            st.subheader("Processed Image:")
            fig, ax = plt.subplots()
            ax.imshow(custom_test_image_processed[0, :, :, 0], cmap='gray')
            ax.set_title(f"Predicted Emotion: {predicted_emotion}")
            ax.axis('off')  # Hide axes
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
