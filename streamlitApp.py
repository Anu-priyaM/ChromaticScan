import streamlit as st
from PIL import Image
import io
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from fastai import *
from fastai.vision import *
from fastai.imports import *
from fastai.vision.all import *
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon=":camera:")

st.title("ChromaticScan")

st.caption(
    "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
)

st.write("Try clicking a leaf image and watch how an AI Model will detect its disease.")

with st.sidebar:
    img = Image.open("./Images/leaf.png")
    st.image(img)
    st.subheader("About ChromaticScan")
    st.write(
        "ChromaticScan is a state-of-the-art convolutional neural network (CNN) algorithm that is specifically designed for detecting plant diseases. It utilizes transfer learning by fine-tuning the ResNet 34 model on a large dataset of leaf images to achieve an impressive 99.2% accuracy in detecting various plant diseases. The algorithm is trained to identify specific patterns and features in the leaf images that are indicative of different types of diseases, such as leaf spots, blights, and wilts."
    )

    st.write(
        "ChromaticScan is designed to be highly robust and accurate, with the ability to detect plant diseases in a wide range of conditions and environments. It can be used to quickly and accurately diagnose plant diseases, allowing farmers and gardeners to take immediate action to prevent the spread of the disease and minimize crop losses. With its high level of accuracy and ease of use, ChromaticScan is poised to revolutionize the way plant diseases are detected and managed in the agricultural industry."
    )

    st.write(
        "The application will infer the one label out of 39 labels, as follows: 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'."
    )

# Class definitions and descriptions
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Background_without_leaves",
]

classes_and_descriptions = {
    "Apple___Apple_scab": "Apple with Apple scab disease detected.",
    "Apple___Black_rot": "Apple with Black rot disease detected.",
    "Apple___Cedar_apple_rust": "Apple with Cedar apple rust disease detected.",
    "Apple___healthy": "Healthy apple leaf detected.",
    "Blueberry___healthy": "Healthy blueberry leaf detected.",
    "Cherry___Powdery_mildew": "Cherry with Powdery mildew disease detected.",
    "Cherry___healthy": "Healthy cherry leaf detected.",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn with Cercospora leaf spot or Gray leaf spot disease detected.",
    "Corn___Common_rust": "Corn with Common rust disease detected.",
    "Corn___Northern_Leaf_Blight": "Corn with Northern Leaf Blight disease detected.",
    "Corn___healthy": "Healthy corn leaf detected.",
    "Grape___Black_rot": "Grape with Black rot disease detected.",
    "Grape___Esca_(Black_Measles)": "Grape with Esca (Black Measles) disease detected.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape with Leaf blight (Isariopsis Leaf Spot) disease detected.",
    "Grape___healthy": "Healthy grape leaf detected.",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange with Haunglongbing (Citrus greening) disease detected.",
    "Peach___Bacterial_spot": "Peach with Bacterial spot disease detected.",
    "Peach___healthy": "Healthy peach leaf detected.",
    "Pepper,_bell___Bacterial_spot": "Bell pepper with Bacterial spot disease detected.",
    "Pepper,_bell___healthy": "Healthy bell pepper leaf detected.",
    "Potato___Early_blight": "Potato with Early blight disease detected.",
    "Potato___Late_blight": "Potato with Late blight disease detected.",
    "Potato___healthy": "Healthy potato leaf detected.",
    "Raspberry___healthy": "Healthy raspberry leaf detected.",
    "Soybean___healthy": "Healthy soybean leaf detected.",
    "Squash___Powdery_mildew": "Squash with Powdery mildew disease detected.",
    "Strawberry___Leaf_scorch": "Strawberry with Leaf scorch disease detected.",
    "Strawberry___healthy": "Healthy strawberry leaf detected.",
    "Tomato___Bacterial_spot": "Tomato leaf with Bacterial spot disease detected.",
    "Tomato___Early_blight": "Tomato leaf with Early blight disease detected.",
    "Tomato___Late_blight": "Tomato leaf with Late blight disease detected.",
    "Tomato___Leaf_Mold": "Tomato leaf with Leaf Mold disease detected.",
    "Tomato___Septoria_leaf_spot": "Tomato leaf with Septoria leaf spot disease detected.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato leaf with Spider mites or Two-spotted spider mite disease detected.",
    "Tomato___Target_Spot": "Tomato leaf with Target Spot disease detected.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf with Tomato Yellow Leaf Curl Virus disease detected.",
    "Tomato___Tomato_mosaic_virus": "Tomato leaf with Tomato mosaic virus disease detected.",
    "Tomato___healthy": "Healthy tomato leaf detected.",
    "Background_without_leaves": "No plant leaf detected in the image.",
}


# Define the functions to load images
def load_uploaded_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image


# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_file_img = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])
elif input_method == "Camera Input":
    camera_file_img = st.camera_input("Take a picture of a leaf...")


# Load the model once
model = load_model('plant_disease_model.h5')


# Define the prediction function
def Plant_Disease_Detection(img_file_path):
    # Load the image and preprocess it
    if isinstance(img_file_path, str):
        img = load_uploaded_image(open(img_file_path, "rb"))
    else:
        img = load_uploaded_image(img_file_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions

    # Make prediction
    predictions = model.predict(img_array)
    idx = np.argmax(predictions)
    probability = predictions[0][idx]
    
    # Prepare confidence message
    confidence_message = f"Confidence: {probability:.2f}" if probability >= 0.6 else "Low confidence in prediction."

    # Retrieve class name and probabilities
    class_name = classes[idx]
    probabilities = predictions[0]  # Get probabilities for all classes

    return class_name, confidence_message, probabilities


# Handling the submission button
submit = st.button(label="Submit Leaf Image")
if submit:
    st.subheader("Output")
    if input_method == "File Uploader":
        img_file_path = uploaded_file_img
    elif input_method == "Camera Input":
        img_file_path = camera_file_img

    prediction, confidence_message, probabilities = Plant_Disease_Detection(img_file_path)
    
    with st.spinner(text="This may take a moment..."):
        st.write(prediction)
        if confidence_message:
            st.write(confidence_message)

        # Visualization
        if probabilities is not None:
            # Ensure probabilities is a numpy array
            if isinstance(probabilities, list):
                probabilities = np.array(probabilities)
            elif hasattr(probabilities, 'numpy'):
                probabilities = probabilities.numpy()

            # Create a DataFrame for better plotting
            prob_df = pd.DataFrame({
                'Class': classes,
                'Probability': probabilities
            })

            # Bar Chart for Probabilities
            plt.figure(figsize=(10, 5))
            sns.barplot(data=prob_df.sort_values('Probability', ascending=False).head(10), x='Probability', y='Class', palette='viridis')
            plt.title("Top 10 Class Probabilities")
            plt.xlabel("Probability")
            plt.ylabel("Class")
            st.pyplot(plt)

            # Pie Chart for the predicted class vs others
            if 0 <= idx < len(probabilities):
                predicted_probability = probabilities[idx]
                other_classes_probability = 1 - predicted_probability
                
                plt.figure(figsize=(7, 7))
                plt.pie(
                    [predicted_probability, other_classes_probability],
                    labels=[prediction, 'Other Classes'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#66c2a5', '#fc8d62']
                )
                plt.title("Prediction Confidence Distribution")
                st.pyplot(plt)
            else:
                st.error("Error: Invalid index for predicted probabilities.")
