import streamlit as st
from PIL import Image
import io
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from fastai.vision.all import *
import pathlib
import matplotlib.pyplot as plt

plt = platform.system()
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

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
        "ChromaticScan is a state-of-the-art convolutional neural network (CNN) algorithm that is specifically designed for detecting plant diseases. "
        "It utilizes transfer learning by fine-tuning the ResNet 34 model on a large dataset of leaf images to achieve an impressive 99.2% accuracy in detecting various plant diseases. "
        "The algorithm is trained to identify specific patterns and features in the leaf images that are indicative of different types of diseases, such as leaf spots, blights, and wilts."
    )
    st.write(
        "ChromaticScan is designed to be highly robust and accurate, with the ability to detect plant diseases in a wide range of conditions and environments. "
        "It can be used to quickly and accurately diagnose plant diseases, allowing farmers and gardeners to take immediate action to prevent the spread of the disease and minimize crop losses."
    )
    st.write(
        "The application will infer the one label out of 39 labels, as follows: 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'."
    )

# Define classes and descriptions
classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", 
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", 
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", 
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", 
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", 
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
    "Background_without_leaves",
]

classes_and_descriptions = {
    # Add descriptions here...
}

# Load image function
def load_uploaded_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image

# Set up sidebar for input method selection
st.subheader("Select Image Input Method")
input_method = st.radio(
    "Options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

# Check input method selected
if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")

# Model file path
export_file_path = "./models/export.pkl"

def Plant_Disease_Detection(img_file):
    model = load_learner(export_file_path)
    prediction, _, outputs = model.predict(img_file)
    confidence = np.max(outputs).item() * 100  # Get confidence percentage
    return prediction, confidence

submit = st.button(label="Submit Leaf Image")
if submit:
    st.subheader("Output")
    if input_method == "File Uploader":
        img_file_path = uploaded_file_img
    elif input_method == "Camera Input":
        img_file_path = camera_file_img

    prediction, confidence = Plant_Disease_Detection(img_file_path)
    treatment = "Refer to local agricultural guidelines for treatment."  # Replace with actual treatment information if available
    
    # Prepare output data
    output_data = {
        "Leaf Name": [img_file_path],
        "Disease/Healthy": [prediction],
        "Confidence (%)": [confidence],
        "Treatment": [treatment],
    }

    # Create DataFrame for display
    output_df = pd.DataFrame(output_data)

    st.write(output_df)

    # Create Pie chart for confidence visualization
    labels = ['Confidence', 'Uncertainty']
    sizes = [confidence, 100 - confidence]
    colors = ['gold', 'lightcoral']
    explode = (0.1, 0)  # explode 1st slice

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    
    st.pyplot(plt)

footer = """
<div style="text-align: center; font-size: medium; margin-top:50px;">
    If you find ChromaticScan useful or interesting, please consider starring it on GitHub.
    <hr>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
