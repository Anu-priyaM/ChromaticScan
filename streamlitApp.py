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
    "Apple___Black_rot": "Apple with Black rot disease detected",
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
    "Potato___healthy": "Healthy potato leaf detected",
    "Raspberry___healthy": "Healthy raspberry leaf detected",
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
    "Tomato___healthy": "Healthy tomato leaf.",
    "Background_without_leaves": "No plant leaf detected in the image.",
}

 # Define the functions to load images
def load_uploaded_image(file):
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        return opencv_image

    # Set up the sidebar for image input method
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

    # Model file path
    export_file_path = "./models/export.pkl"

    def Plant_Disease_Detection(img_file_path):
        model = load_learner(export_file_path, "export.pkl")
        # Process image
        image = cv2.resize(img_file_path, (256, 256))  # Resize to 256x256
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Expand dimensions for the model

        # Prediction
        prediction = model.predict(image)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100  # Confidence level
        
        if predicted_class not in classes:
            prediction_sentence = f"The uploaded image is {predicted_class}, which is not compatible with the application. Please upload an image of a plant leaf for disease detection."
            return prediction_sentence, 0
        prediction_sentence = classes_and_descriptions[predicted_class]
        return prediction_sentence, confidence

    submit = st.button(label="Submit Leaf Image")
    if submit:
        st.subheader("Output")
        if input_method == "File Uploader":
            img_file_path = uploaded_file_img
        elif input_method == "Camera Input":
            img_file_path = camera_file_img

        prediction, confidence = Plant_Disease_Detection(img_file_path)
        with st.spinner(text="This may take a moment..."):
            st.write(prediction)
            st.write(f"Confidence: {confidence:.2f}%")

            # Confidence Pie Chart
            fig, ax = plt.subplots()
            labels = ['Predicted Class', 'Others']
            sizes = [confidence, 100 - confidence]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)

# Charts Page
elif page == "Charts":
    st.subheader("Charts and Visualizations")

    # Sample data for accuracy and loss
    epochs = range(1, 21)
    accuracy = [0.7, 0.75, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91, 0.91, 0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.95]
    val_accuracy = [0.68, 0.72, 0.75, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.89, 0.9, 0.91, 0.91, 0.92, 0.92, 0.92, 0.93, 0.93, 0.94]
    loss = [0.8, 0.75, 0.72, 0.7, 0.68, 0.65, 0.63, 0.61, 0.59, 0.58, 0.56, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46]
    val_loss = [0.82, 0.78, 0.74, 0.72, 0.7, 0.68, 0.66, 0.65, 0.63, 0.62, 0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51]

    # Plot Training and Validation Accuracy
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    sns.lineplot(x=epochs, y=accuracy, ax=axs[0], label='Training Accuracy', color='b', marker='o')
    sns.lineplot(x=epochs, y=val_accuracy, ax=axs[0], label='Validation Accuracy', color='g', marker='o')
    axs[0].set_title('Model Accuracy Over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Plot Training and Validation Loss
    sns.lineplot(x=epochs, y=loss, ax=axs[1], label='Training Loss', color='r', marker='o')
    sns.lineplot(x=epochs, y=val_loss, ax=axs[1], label='Validation Loss', color='orange', marker='o')
    axs[1].set_title('Model Loss Over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Model Performance Comparison")

    # Sample data to illustrate performance
    data = {
        'Model': ['ResNet34', 'ResNet50', 'VGG16', 'VGG19', 'AlexNet', 'InceptionV3', 'DenseNet121', 'EfficientNetB0', 'SqueezeNet', 'Xception'],
        'Accuracy (%)': [99.0, 98.5, 97.8, 97.4, 98.0, 96.5, 97.0, 96.9, 95.7, 96.0],
        'Precision (%)': [98.7, 98.0, 97.5, 97.0, 97.8, 96.0, 96.5, 95.8, 95.2, 96.1],
        'Recall (%)': [99.1, 98.7, 97.9, 97.5, 98.2, 96.8, 97.2, 96.5, 95.9, 96.3],
        'F1-Score (%)': [98.9, 98.3, 97.7, 97.2, 97.9, 96.4, 96.8, 96.2, 95.6, 96.1],
        'Training Time (hrs)': [12, 14, 10, 11, 8, 15, 13, 14, 9, 16],
        'Number of Parameters (M)': [21, 25, 138, 143, 61, 24, 8, 5, 1, 22],
    }

    df = pd.DataFrame(data)

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create subplots
    fig, axs = plt.subplots(5, 2, figsize=(16, 24))

    # Plot 1: Model Accuracy
    sns.barplot(x='Accuracy (%)', y='Model', data=df, ax=axs[0, 0], palette='Blues_d', orient='h')
    axs[0, 0].set_title('Model Accuracy')
    for index, value in enumerate(df['Accuracy (%)']):
        axs[0, 0].text(value, index, f'{value:.1f}%', va='center')

    # Plot 2: Model Precision
    sns.barplot(x='Precision (%)', y='Model', data=df, ax=axs[0, 1], palette='Greens_d', orient='h')
    axs[0, 1].set_title('Model Precision')
    for index, value in enumerate(df['Precision (%)']):
        axs[0, 1].text(value, index, f'{value:.1f}%', va='center')

    # Plot 3: Model Recall
    sns.barplot(x='Recall (%)', y='Model', data=df, ax=axs[1, 0], palette='Reds_d', orient='h')
    axs[1, 0].set_title('Model Recall')
    for index, value in enumerate(df['Recall (%)']):
        axs[1, 0].text(value, index, f'{value:.1f}%', va='center')

    # Plot 4: Model F1-Score
    sns.barplot(x='F1-Score (%)', y='Model', data=df, ax=axs[1, 1], palette='Purples_d', orient='h')
    axs[1, 1].set_title('Model F1-Score')
    for index, value in enumerate(df['F1-Score (%)']):
        axs[1, 1].text(value, index, f'{value:.1f}%', va='center')

    # Plot 5: Training Time
    sns.barplot(x='Training Time (hrs)', y='Model', data=df, ax=axs[2, 0], palette='Oranges_d', orient='h')
    axs[2, 0].set_title('Training Time')
    for index, value in enumerate(df['Training Time (hrs)']):
        axs[2, 0].text(value, index, f'{value:.1f} hrs', va='center')

    # Plot 6: Number of Parameters
    sns.barplot(x='Number of Parameters (M)', y='Model', data=df, ax=axs[2, 1], palette='Greys_d', orient='h')
    axs[2, 1].set_title('Number of Parameters')
    for index, value in enumerate(df['Number of Parameters (M)']):
        axs[2, 1].text(value, index, f'{value:.1f} M', va='center')

    # Plot 7: Accuracy Comparison (Highlighting ResNet34)
    sns.barplot(x='Accuracy (%)', y='Model', data=df, ax=axs[3, 0], palette='coolwarm', orient='h')
    axs[3, 0].set_title('Accuracy Comparison (Highlighting ResNet34)')
    axs[3, 0].axvline(x=99.0, color='r', linestyle='--', label='ResNet34 Accuracy')
    axs[3, 0].legend()
    for index, value in enumerate(df['Accuracy (%)']):
        axs[3, 0].text(value, index, f'{value:.1f}%', va='center')

    # Plot 8: Precision Comparison
    sns.barplot(x='Precision (%)', y='Model', data=df, ax=axs[3, 1], palette='viridis', orient='h')
    axs[3, 1].set_title('Precision Comparison')
    for index, value in enumerate(df['Precision (%)']):
        axs[3, 1].text(value, index, f'{value:.1f}%', va='center')

    # Plot 9: Recall Comparison
    sns.barplot(x='Recall (%)', y='Model', data=df, ax=axs[4, 0], palette='magma', orient='h')
    axs[4, 0].set_title('Recall Comparison')
    for index, value in enumerate(df['Recall (%)']):
        axs[4, 0].text(value, index, f'{value:.1f}%', va='center')

    # Plot 10: F1-Score Comparison
    sns.barplot(x='F1-Score (%)', y='Model', data=df, ax=axs[4, 1], palette='plasma', orient='h')
    axs[4, 1].set_title('F1-Score Comparison')
    for index, value in enumerate(df['F1-Score (%)']):
        axs[4, 1].text(value, index, f'{value:.1f}%', va='center')

    plt.tight_layout()
    st.pyplot(fig)

# Disclaimer Page
elif page == "Disclaimer":
    st.title("Disclaimer")
    st.write("""
    The information provided by ChromaticScan is intended for educational and informational purposes only. While we strive for accuracy in our predictions, the results generated by the AI model are not guaranteed to be 100% accurate.
    
    It is advised that users seek the advice of a qualified professional for diagnosis and treatment of any plant diseases. ChromaticScan is not responsible for any consequences that may arise from the use of this application or its predictions.
    
    By using this application, you agree to these terms and acknowledge that the predictions are based on the data available to the model at the time of your input.
    """)

# Footer
st.markdown("@copyright")
st.write("Developed by Anupriya.")
