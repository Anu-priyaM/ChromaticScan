import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from fastai.learner import load_learner
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the sidebar
st.set_page_config(page_title="ChromaticScan", page_icon=":camera:")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Home", "Prediction", "Charts", "Disclaimer"])

# Home Page
if page == "Home":
    st.write(
        "Welcome to ChromaticScan, your AI-powered solution for detecting plant diseases. "
        "Use the sidebar to navigate to the prediction or charts sections."
    )
    st.subheader("Benefits of Plant Disease Prediction")
    st.write("""
    - *Early Detection*: Identifying diseases at an early stage helps in timely intervention, preventing widespread damage.
    - *Cost-Effective*: Early treatment reduces the need for extensive use of pesticides and other treatments, saving costs.
    - *Increased Yield*: Healthy plants result in better yield and quality, ensuring profitability for farmers.
    - *Data-Driven Decisions*: Use of AI and machine learning provides insights that can guide agricultural practices and strategies.
    """)

    st.subheader("Usage")
    st.write("""
    - *Upload or capture a leaf image*: Use the app to upload an image of a plant leaf or take a picture using the camera.
    - *Receive diagnosis and recommendations*: The app will predict the disease and provide recommendations for treatment.
    - *Monitor and manage*: Regular use of the app can help in monitoring plant health and managing diseases effectively.
    """)

    st.subheader("Machine Learning Algorithm")
    st.write("""
    - *ResNet 34*: ChromaticScan uses a deep learning model based on ResNet 34, a type of convolutional neural network.
    - *Transfer Learning*: The model is fine-tuned using a dataset of plant leaf images, leveraging pre-trained weights for improved accuracy.
    - *High Accuracy*: The model achieves an accuracy of 99.2%, capable of distinguishing between 39 different classes of plant diseases and healthy leaves.
    """)

# Prediction Page
elif page == "Prediction":
    st.title("ChromaticScan")
    st.caption(
        "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
    )

    st.write("Try clicking a leaf image and watch how an AI Model will detect its disease.")
    st.write(
        "The application will infer one label out of 39 labels, as follows: "
        "'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy', "
        "'Background_without_leaves', 'Blueberry__healthy', 'Cherry__healthy', "
        "'Corn__Cercospora_leaf_spot Gray_leaf_spot', 'Corn_Common_rust', 'Corn__Northern_Leaf_Blight', "
        "'Corn__healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', "
        "'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', "
        "'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell__Bacterial_spot', "
        "'Pepper,bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato__healthy', "
        "'Raspberry__healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry__Leaf_scorch', "
        "'Strawberry__healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight', "
        "'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite', "
        "'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus', "
        "'Tomato___healthy'."
    )

    classes = [
        "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__healthy",
        "Blueberry__healthy", "Cherry_healthy", "Corn__Cercospora_leaf_spot Gray_leaf_spot",
        "Corn__Common_rust", "Corn_Northern_Leaf_Blight", "Corn_healthy", "Grape__Black_rot",
        "Grape__Esca(Black_Measles)", "Grape__Leaf_blight(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange__Haunglongbing(Citrus_greening)", "Peach__Bacterial_spot", "Peach__healthy",
        "Pepper,bell_Bacterial_spot", "Pepper,_bell_healthy", "Potato__Early_blight",
        "Potato__Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean__healthy",
        "Squash__Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry__healthy",
        "Tomato__Bacterial_spot", "Tomato_Early_blight", "Tomato__Late_blight",
        "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot", 
        "Tomato__Spider_mites Two-spotted_spider_mite", "Tomato__Target_Spot", 
        "Tomato__Tomato_Yellow_Leaf_Curl_Virus", "Tomato__Tomato_mosaic_virus", 
        "Tomato___healthy", "Background_without_leaves",
    ]

    classes_and_descriptions = {
        "Apple___Apple_scab": "Apple with Apple scab disease detected.",
        "Apple___Black_rot": "Apple with Black rot disease detected.",
        "Apple___Cedar_apple_rust": "Apple with Cedar apple rust disease detected.",
        "Apple___healthy": "Healthy apple leaf detected.",
        "Blueberry___healthy": "Healthy blueberry leaf detected.",
        "Cherry___healthy": "Healthy cherry leaf detected.",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn with Cercospora leaf spot or Gray leaf spot disease detected.",
        "Corn___Common_rust": "Corn with Common rust disease detected.",
        "Corn___Northern_Leaf_Blight": "Corn with Northern Leaf Blight disease detected.",
        "Corn___healthy": "Healthy corn leaf detected.",
        "Grape___Black_rot": "Grape with Black rot disease detected.",
        "Grape__Esca(Black_Measles)": "Grape with Esca (Black Measles) disease detected.",
        "Grape__Leaf_blight(Isariopsis_Leaf_Spot)": "Grape with Leaf blight (Isariopsis Leaf Spot) disease detected.",
        "Grape___healthy": "Healthy grape leaf detected.",
        "Orange__Haunglongbing(Citrus_greening)": "Orange with Haunglongbing (Citrus greening) disease detected.",
        "Peach___Bacterial_spot": "Peach with Bacterial spot disease detected.",
        "Peach___healthy": "Healthy peach leaf detected.",
        "Pepper,bell__Bacterial_spot": "Bell pepper with Bacterial spot disease detected.",
        "Pepper,bell__healthy": "Healthy bell pepper leaf detected.",
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
        "Background_without_leaves": "No leaves detected in the image."
    }

    # Upload image for prediction
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for the model
        model_input = np.array(image.resize((224, 224))) / 255.0  # Resize and normalize
        model_input = np.expand_dims(model_input, axis=0)

        # Load the model (ensure the model path is correct)
        model = load_model('models/export.h5')

        # Make prediction
        prediction = model.predict(model_input)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_class_index]
        description = classes_and_descriptions[predicted_class]

        st.write(f"Predicted Disease: *{predicted_class}*")
        st.write(description)

# Charts Page
elif page == "Charts":
    st.title("Analysis of Plant Diseases")

    # Load data (ensure the CSV path is correct)
    data = pd.read_csv('path/to/your/data.csv')

    st.subheader("Disease Distribution")
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x='Disease', order=data['Disease'].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader("Disease Severity Over Time")
    # Assuming you have a 'Date' and 'Severity' columns in your data
    data['Date'] = pd.to_datetime(data['Date'])
    severity_over_time = data.groupby('Date').mean()['Severity']
    plt.figure(figsize=(10, 5))
    plt.plot(severity_over_time.index, severity_over_time.values)
    plt.title("Average Disease Severity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Severity")
    st.pyplot(plt)

# Disclaimer Page
elif page == "Disclaimer":
    st.title("Disclaimer")
    st.write("""
    The predictions made by this application are based on a machine learning model and are intended for informational purposes only. 
    Users should consult with agricultural experts before making decisions based on the results.
    The developers are not responsible for any consequences resulting from the use of this application.
    """)
