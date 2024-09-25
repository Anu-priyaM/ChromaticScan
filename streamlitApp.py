import streamlit as st
from PIL import Image
import io
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from fastai import *
from fastai.vision import *
from fastai.imports import *
from fastai.vision.all import *
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import platform

# Adjusting the path for Windows
plt = platform.system()
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon=":camera:")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "Charts", "Disclaimer"])

# Home Page
if page == "Home":
    st.write(
        "Welcome to ChromaticScan, your AI-powered solution for detecting plant diseases. "
        "Use the sidebar to navigate to the prediction or charts sections."
    )
    st.subheader("Benefits of Plant Disease Prediction")
    st.write(""" 
    - **Early Detection**: Identifying diseases at an early stage helps in timely intervention, preventing widespread damage.
    - **Cost-Effective**: Early treatment reduces the need for extensive use of pesticides and other treatments, saving costs.
    - **Increased Yield**: Healthy plants result in better yield and quality, ensuring profitability for farmers.
    - **Data-Driven Decisions**: Use of AI and machine learning provides insights that can guide agricultural practices and strategies.
    """)

    st.subheader("Usage")
    st.write(""" 
    - **Upload or capture a leaf image**: Use the app to upload an image of a plant leaf or take a picture using the camera.
    - **Receive diagnosis and recommendations**: The app will predict the disease and provide recommendations for treatment.
    - **Monitor and manage**: Regular use of the app can help in monitoring plant health and managing diseases effectively.
    """)

    st.subheader("Machine Learning Algorithm")
    st.write(""" 
    - **ResNet 34**: ChromaticScan uses a deep learning model based on ResNet 34, a type of convolutional neural network.
    - **Transfer Learning**: The model is fine-tuned using a dataset of plant leaf images, leveraging pre-trained weights for improved accuracy.
    - **High Accuracy**: The model achieves an accuracy of 99.2%, capable of distinguishing between 39 different classes of plant diseases and healthy leaves.
    """)

# Prediction Page
elif page == "Prediction":
    st.title("ChromaticScan")
    st.caption(
        "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
    )

    st.write("Try clicking a leaf image and watch how an AI Model will detect its disease.")
    st.write(
        "The application will infer the one label out of 39 labels, as follows: 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'."
    )

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
        "Tomato___Leaf_Mold": "Tomato with Leaf Mold disease detected.",
        "Tomato___Septoria_leaf_spot": "Tomato leaf with Septoria leaf spot disease detected.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato leaf with Spider mites Two-spotted spider mite disease detected.",
        "Tomato___Target_Spot": "Tomato with Target Spot disease detected.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato with Yellow Leaf Curl Virus disease detected.",
        "Tomato___Tomato_mosaic_virus": "Tomato with mosaic virus disease detected.",
        "Tomato___healthy": "Healthy tomato leaf detected.",
        "Background_without_leaves": "Background without any leaves detected.",
    }

    # Load the model
    model = load_model('path_to_your_model.h5')

    # File uploader
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Prepare the image for prediction
        img = load_img(uploaded_file, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = classes[predicted_class_index]

        # Display the predicted class and its description
        st.write(f"Prediction: {predicted_class}")
        st.write(classes_and_descriptions[predicted_class])

# Charts Page
elif page == "Charts":
    st.title("Data Visualization")
    st.write("Visualizing data to better understand plant diseases.")

    # Sample data visualization using seaborn and matplotlib
    sample_data = pd.DataFrame({
        'Disease': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
        'Count': [10, 5, 2, 20]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sample_data, x='Disease', y='Count', palette='viridis')
    plt.title('Sample Count of Diseases Detected')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# Disclaimer Page
elif page == "Disclaimer":
    st.title("Disclaimer")
    st.write(""" 
    The information provided by ChromaticScan is for educational and informational purposes only. 
    The application is not a substitute for professional agricultural advice or treatment. 
    Always consult with a qualified agricultural expert or plant pathologist for accurate diagnosis and treatment options.
    """)

st.markdown("---")
st.write("© 2024 ChromaticScan. All rights reserved.")
