
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import sklearn
import lightgbm
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np


# Set page configuration

st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")


kidney_disease_model = tf.keras.models.load_model(r'C:\Users\lohit\OneDrive\Desktop\kid\saved model\Kidneydisease.h5')


with st.sidebar:
    selected = option_menu('Kidney Disease Prediction System',
                           ['Information', 'Prediction', 'Remedies'],
                          menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)




def preprocess_image(image):
    pic = []
    # Read the image using PIL
    img_pil = Image.open(image)
    
    # Convert PIL image to NumPy array
    img_np = np.array(img_pil)
    
    # Resize the image
    img_resized = cv2.resize(img_np, (28, 28))
    
    # Convert the image to RGB (OpenCV uses BGR by default)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Display the resized image
    st.image(img_rgb, caption='Resized Image', use_column_width=True)
    
    # Append the resized image to a list
    pic.append(img_rgb)
    
    # Convert the list to a NumPy array
    pic1 = np.array(pic)
    return pic1

# Disease categories
disease_categories = ['Cyst', 'Normal', 'Stone', 'Tumor']


if selected == "Information":
        st.title("Kidney Disease Information")
        # Add your disease information content here
        st.write("**Chronic kidney disease (CKD) is a long-term condition where the kidneys don't work as well as they should.**")
        st.write("**Common causes include:**")
        st.write("- High blood pressure")
        st.write("- Diabetes")
        st.write("- Glomerulonephritis (inflammation of the kidney's filtering units)")
        st.write("- Polycystic kidney disease (an inherited condition)")
    # Add more information as needed
elif selected == "Prediction":
        st.title("Kidney Disease Prediction")
        model =kidney_disease_model

        st.write("Upload a medical image to detect chronic kidney disease.")

        uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

            if st.button('Predict'):
                with st.spinner('Predicting...'):
                    processed_image = preprocess_image(uploaded_file)
                    prediction = model.predict(processed_image)
                    predicted_class_index = np.argmax(prediction, axis=1)[0]
                    predicted_class = disease_categories[predicted_class_index]
                    confidence = prediction[0][predicted_class_index] * 100
                    st.success(f'Prediction: {predicted_class}')

elif selected == "Remedies":
        st.title("Kidney Disease Remedies")
        # Add your remedies content here
        st.write("**Remedies for kidney disease depend on the stage of the disease and the underlying cause.**")
        st.write("**General approaches include:**")
        st.write("- **Blood pressure control:** Maintaining healthy blood pressure is essential to slow the progression of CKD.")
        st.write("- **Blood sugar control:** For people with diabetes, managing blood sugar levels can help prevent kidney damage.")
        st.write("- **Dietary changes:** A healthy diet low in sodium, potassium, and phosphorus can help manage symptoms and slow the progression of CKD.")
        link1 = "https://economictimes.indiatimes.com/news/how-to/7-effective-natural-ways-that-will-keep-your-kidney-healthy/articleshow/88254841.cms?from=mdr"  # Replace with your actual link
        st.write(f"*For more information:[Click here]({link1})")       
        # Add more remedies as needed
