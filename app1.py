import streamlit as st
import os
import shutil
from PIL import Image
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('imageclassifier_44mb.h5')

# Function to classify image as meme or normal photo
def classify_image(image):
    # Preprocess image
    img_array = np.array(image.resize((256, 256))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Perform classification
    prediction = model.predict(img_array)
    
    # Classify as meme if prediction score is > 0.5
    label = 'Normal Photo' if prediction > 0.5 else 'Meme'
    
    return label

# Function to delete meme
def delete_meme(image_path):
    # Delete image from storage
    os.remove(image_path)
    st.success('Meme deleted successfully!')

# Streamlit UI
st.title('Image Classification: Meme or Normal Photo')

# Upload images
uploaded_files = st.file_uploader("Upload multiple images...", type=['jpg', 'png'], accept_multiple_files=True)

# Process uploaded images
if uploaded_files:
    # Create directories if not exist
    os.makedirs('memes', exist_ok=True)
    os.makedirs('regular', exist_ok=True)
    
    # Classify and move uploaded images
    for uploaded_file in uploaded_files:
        # Classify image
        image = Image.open(uploaded_file)
        label = classify_image(image)

        # Move images to appropriate folder based on classification
        if label == 'Meme':
            dest_dir = 'memes'
        else:
            dest_dir = 'regular'

        # Move file
        with open(os.path.join(dest_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Delete the uploaded file
        uploaded_file.close()

# Display images and delete options
regular_images = os.listdir('regular') if os.path.exists('regular') else []
memes_images = os.listdir('memes') if os.path.exists('memes') else []

if regular_images:
    st.header('Regular Photos:')
    for img_name in regular_images:
        st.image(f'regular/{img_name}', use_column_width=True)

    if st.button('Delete All Regular Image'):
        for img_name in regular_images:
            delete_meme(f'regular/{img_name}')


if memes_images:
    st.header('Memes:')
    for img_name in memes_images:
        st.image(f'memes/{img_name}', use_column_width=True)

    # Option to delete all memes
    if st.button('Delete All Memes'):
        for img_name in memes_images:
            delete_meme(f'memes/{img_name}')
