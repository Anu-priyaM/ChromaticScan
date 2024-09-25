import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import platform

# Platform-specific path handling
plt_platform = platform.system()
if plt_platform == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon="🌿")

st.title("ChromaticScan")

st.caption(
    "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
)

# Sidebar for navigation
with st.sidebar:
    img = Image.open("./Images/leaf.png")
    st.image(img)
    st.subheader("Navigation")
    page = st.radio("Go to", ["Home", "Prediction", "Charts"])

# Classes and descriptions
classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
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
    "Cherry_(including_sour)___Powdery_mildew": "Cherry with Powdery mildew disease detected.",
    "Cherry_(including_sour)___healthy": "Healthy cherry leaf detected.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn with Cercospora leaf spot or Gray leaf spot disease detected.",
    "Corn_(maize)___Common_rust_": "Corn with Common rust disease detected.",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn with Northern Leaf Blight disease detected.",
    "Corn_(maize)___healthy": "Healthy corn leaf detected.",
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

# Define remedies for each class
remedies = {
    "Apple___Apple_scab": "Apply fungicide and remove affected leaves.",
    "Apple___Black_rot": "Use copper fungicides and ensure good air circulation.",
    "Apple___Cedar_apple_rust": "Use resistant varieties and apply fungicides.",
    "Apple___healthy": "No treatment needed.",
    "Blueberry___healthy": "No treatment needed.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply fungicides to manage powdery mildew.",
    "Cherry_(including_sour)___healthy": "No treatment needed.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides and improve drainage.",
    "Corn_(maize)___Common_rust_": "Use resistant hybrids and fungicides.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops and use resistant varieties.",
    "Corn_(maize)___healthy": "No treatment needed.",
    "Grape___Black_rot": "Use fungicides and remove infected fruits.",
    "Grape___Esca_(Black_Measles)": "Prune affected vines and apply appropriate fungicides.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides and improve air circulation.",
    "Grape___healthy": "No treatment needed.",
    "Orange___Haunglongbing_(Citrus_greening)": "Use disease-free plants and manage insect vectors.",
    "Peach___Bacterial_spot": "Apply copper-based fungicides and improve air circulation.",
    "Peach___healthy": "No treatment needed.",
    "Pepper,_bell___Bacterial_spot": "Use resistant varieties and apply copper fungicides.",
    "Pepper,_bell___healthy": "No treatment needed.",
    "Potato___Early_blight": "Use fungicides and rotate crops.",
    "Potato___Late_blight": "Use resistant varieties and apply fungicides.",
    "Potato___healthy": "No treatment needed.",
    "Raspberry___healthy": "No treatment needed.",
    "Soybean___healthy": "No treatment needed.",
    "Squash___Powdery_mildew": "Apply fungicides and improve air circulation.",
    "Strawberry___Leaf_scorch": "Water adequately and apply fungicides.",
    "Strawberry___healthy": "No treatment needed.",
    "Tomato___Bacterial_spot": "Use resistant varieties and apply copper fungicides.",
    "Tomato___Early_blight": "Apply fungicides and rotate crops.",
    "Tomato___Late_blight": "Use resistant varieties and apply fungicides.",
    "Tomato___Leaf_Mold": "Improve ventilation and apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Rotate crops and use resistant varieties.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticides and improve air circulation.",
    "Tomato___Target_Spot": "Use fungicides and rotate crops.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Manage aphids and use resistant varieties.",
    "Tomato___Tomato_mosaic_virus": "Use virus-free seeds and manage insect vectors.",
    "Tomato___healthy": "No treatment needed.",
    "Background_without_leaves": "No treatment needed.",
}

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model_file(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")

model_path = "Plant_disease.h5"
model = load_model_file(model_path)

# Function to load the uploaded image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Plant Disease Detection function
def Plant_Disease_Detection(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 224, 224, 3))
    
    # Get prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    return predicted_class, confidence

# Page routing logic
if page == "Home":
    st.header("Welcome to ChromaticScan!")
    st.write("This app helps you detect diseases in plant leaves. "
             "Simply upload an image of a leaf, and the app will classify the disease and provide treatment recommendations.")
    
elif page == "Prediction":
    st.header("Upload a Plant Leaf Image")
    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if image_file is not None:
        image = load_image(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        predicted_class, confidence = Plant_Disease_Detection(image)
        
        # Display prediction results
        st.subheader("Prediction Result:")
        st.write(f"**Predicted Class:** {predicted_class.replace('_', ' ')}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        # Treatment recommendations
        st.subheader("Treatment Recommendations:")
        if predicted_class in remedies:
            st.write(remedies[predicted_class])
        else:
            st.write("No treatment recommendations available.")
        
        # Show confidence pie chart
        st.subheader("Confidence Distribution")
        fig, ax = plt.subplots()
        ax.pie([confidence, 100 - confidence], labels=['Predicted Confidence', 'Other'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

elif page == "Charts":
    st.header("Model Performance Comparison")
    model_names = ["ResNet34", "InceptionV3", "VGG16", "EfficientNet"]
    accuracy = [0.992, 0.970, 0.950, 0.960]
    precision = [0.98, 0.95, 0.93, 0.94]
    recall = [0.97, 0.96, 0.91, 0.92]
    f1_score = [0.975, 0.955, 0.92, 0.93]

    data = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score
    })

    # Accuracy comparison
    st.subheader("Model Accuracy Comparison")
    sns.barplot(data=data, x='Model', y='Accuracy', palette='viridis')
    st.pyplot(plt.gcf())

    # Precision comparison
    st.subheader("Model Precision Comparison")
    sns.barplot(data=data, x='Model', y='Precision', palette='viridis')
    st.pyplot(plt.gcf())

    # Recall comparison
    st.subheader("Model Recall Comparison")
    sns.barplot(data=data, x='Model', y='Recall', palette='viridis')
    st.pyplot(plt.gcf())

    # F1 Score comparison
    st.subheader("Model F1-Score Comparison")
    sns.barplot(data=data, x='Model', y='F1-Score', palette='viridis')
    st.pyplot(plt.gcf())
