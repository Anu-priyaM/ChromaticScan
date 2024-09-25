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

plt = platform.system()
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon=":camera:")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Plant Disease Detection", "Disclaimer"])

if page == "Home":
    st.title("Welcome to ChromaticScan!")
    st.write(
        "ChromaticScan is a state-of-the-art convolutional neural network (CNN) algorithm specifically designed for detecting plant diseases. It utilizes transfer learning by fine-tuning the ResNet 34 model on a large dataset of leaf images to achieve an impressive 99.2% accuracy in detecting various plant diseases."
    )
    st.image("./Images/leaf.png", caption="Plant Leaf")
    st.write("Try clicking the 'Plant Disease Detection' page to analyze your plant leaf images.")

elif page == "Disclaimer":
    st.title("Disclaimer")
    st.write(
        "The information provided by ChromaticScan is intended for informational purposes only and should not be considered as medical or agricultural advice. Users are encouraged to consult with agricultural professionals for accurate diagnosis and treatment of plant diseases. While the application is designed to provide reliable predictions, the accuracy may vary based on factors such as image quality and environmental conditions."
    )
    st.write("Use the application at your own risk. The developers are not responsible for any losses or damages arising from the use of the application.")

else:
    st.title("ChromaticScan: Plant Disease Detection")
    st.caption(
        "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
    )
    st.write("Try clicking a leaf image and watch how an AI Model will detect its disease.")

    with st.sidebar:
        img = Image.open("./Images/leaf.png")
        st.image(img)
        st.subheader("About ChromaticScan")
        st.write(
            "ChromaticScan is a state-of-the-art convolutional neural network (CNN) algorithm that is specifically designed for detecting plant diseases. It utilizes transfer learning by fine-tuning the ResNet 34 model on a large dataset of leaf images to achieve an impressive 99.2% accuracy in detecting various plant diseases."
        )

        st.write(
            "ChromaticScan is designed to be highly robust and accurate, with the ability to detect plant diseases in a wide range of conditions and environments. It can be used to quickly and accurately diagnose plant diseases, allowing farmers and gardeners to take immediate action to prevent the spread of the disease and minimize crop losses. With its high level of accuracy and ease of use, ChromaticScan is poised to revolutionize the way plant diseases are detected and managed in the agricultural industry."
        )

        st.write(
            "The application will infer the one label out of 39 labels, as follows: 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'."
        )

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
        "Tomato___Bacterial_spot": "Tomato with Bacterial spot disease detected.",
        "Tomato___Early_blight": "Tomato with Early blight disease detected.",
        "Tomato___Late_blight": "Tomato with Late blight disease detected.",
        "Tomato___Leaf_Mold": "Tomato with Leaf Mold disease detected.",
        "Tomato___Septoria_leaf_spot": "Tomato with Septoria leaf spot disease detected.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato with Spider mites (Two-spotted spider mite) disease detected.",
        "Tomato___Target_Spot": "Tomato with Target Spot disease detected.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato with Tomato Yellow Leaf Curl Virus disease detected.",
        "Tomato___Tomato_mosaic_virus": "Tomato with Tomato mosaic virus disease detected.",
        "Tomato___healthy": "Healthy tomato leaf detected.",
        "Background_without_leaves": "Background without leaves detected.",
    }

    # Load model
    model = load_learner('./models/export.pkl', 'export.pkl')

    # Image upload
    st.subheader("Select Image Input Method")
    input_method = st.radio("Options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

    # Check which input method was selected
    if input_method == "File Uploader":
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Load and display the uploaded image
            uploaded_file_img = load_uploaded_image(uploaded_file)
            st.image(uploaded_file_img, caption="Uploaded Image", width=300)
            st.success("Image uploaded successfully!")
        else:
            st.warning("Please upload an image file.")

    elif input_method == "Camera Input":
        st.warning("Please allow access to your camera.")
        camera_image_file = st.camera_input("Click an Image")
        if camera_image_file is not None:
            # Load and display the camera input image
            camera_file_img = load_uploaded_image(camera_image_file)
            st.image(camera_file_img, caption="Camera Input Image", width=300)
            st.success("Image clicked successfully!")
        else:
            st.warning("Please click an image.")

    submit = st.button(label="Submit Leaf Image")
    if submit:
        st.subheader("Output")
        if input_method == "File Uploader":
            img_file_path = uploaded_file_img
        elif input_method == "Camera Input":
            img_file_path = camera_file_img

        prediction, confidence_message = Plant_Disease_Detection(img_file_path)
        with st.spinner(text="This may take a moment..."):
            st.write(prediction)
            if confidence_message:
                st.write(confidence_message)

footer = """
<div style="text-align: center; font-size: medium; margin-top:50px;">
    If you find ChromaticScan useful or interesting, please consider starring it on GitHub.
    <hr>
    <a href="https://github.com/SaiJeevanPuchakayala/ChromaticScan" target="_blank">
    <img src="https://img.shields.io/github/stars/SaiJeevanPuchakayala/ChromaticScan.svg?style=social" alt="GitHub stars">
  </a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
