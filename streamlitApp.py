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
        "ChromaticScan is designed to be highly robust and accurate, with the ability to detect plant diseases in a wide range of conditions and environments. It can be used to quickly and accurately diagnose plant diseases, allowing farmers and gardeners to take immediate action to prevent the spread of the disease and minimize crop losses."
    )

    st.write(
        "The application will infer one label out of 39 labels, including diseases like Apple scab, Early blight, Late blight, and more."
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
    # (Add your plant disease descriptions here)
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

# Define remedies for each class
remedies = {
    "Apple___Apple_scab": "Apply fungicide and remove affected leaves.",
    "Apple___Black_rot": "Use copper fungicides and ensure good air circulation.",
    "Apple___Cedar_apple_rust": "Use resistant varieties and apply fungicides.",
    "Apple___healthy": "No treatment needed.",
    "Blueberry___healthy": "No treatment needed.",
    "Cherry___Powdery_mildew": "Apply fungicides to manage powdery mildew.",
    "Cherry___healthy": "No treatment needed.",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Apply fungicides and improve drainage.",
    "Corn___Common_rust": "Use resistant hybrids and fungicides.",
    "Corn___Northern_Leaf_Blight": "Rotate crops and use resistant varieties.",
    "Corn___healthy": "No treatment needed.",
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

# Define the functions to load images
def load_uploaded_image(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    return opencv_image

# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# model file path
export_file_path = "./models/export.pkl"

def Plant_Disease_Detection(img_file_path):
    model = load_learner(export_file_path, "export.pkl")
    # Get prediction and confidence score
    prediction, idx, probabilities = model.predict(img_file_path)
    
    # Extract the confidence score
    confidence_score = probabilities[idx].item()

    # Check if prediction is valid
    if prediction not in classes:
        prediction_sentence = f"The uploaded image is {prediction}, which is not compatible with the application. Please upload an image of a plant leaf for disease detection."
        return prediction_sentence, None, None, None

    # Generate prediction message with confidence score
    prediction_sentence = classes_and_descriptions[prediction]
    confidence_message = f"Confidence: {confidence_score * 100:.2f}%"
    
    # Get remedy for the detected disease
    remedy = remedies.get(prediction, "No remedy available for this disease.")
    
    return prediction_sentence, confidence_message, probabilities, remedy

submit = st.button(label="Submit Leaf Image")
if submit:
    st.subheader("Output")
    if input_method == "File Uploader":
        img_file_path = uploaded_file_img
    elif input_method == "Camera Input":
        img_file_path = camera_file_img

    prediction, confidence_message, probabilities, remedy = Plant_Disease_Detection(img_file_path)
    
    with st.spinner(text="This may take a moment..."):
        st.write(prediction)
        if confidence_message:
            st.write(confidence_message)
        if remedy:
            st.write(f"Recommended Treatment: {remedy}")

        # Visualization
        if probabilities is not None:
            # Convert probabilities to a numpy array
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

footer = """
<div style="text-align: center; font-size: medium; margin-top:50px;">
   This is a final year project developed by Anupriya.
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
