import os
from tensorflow.keras.models import load_model
from cv2 import imread, resize
from shutil import move
import numpy as np

# Load model
model = load_model('imageclassifier_44mb.h5')

# Directory containing images to classify
dir = 'data'

# Create directories for regular and meme images if they don't exist
regular_dir = 'regular'
memes_dir = 'memes'
os.makedirs(regular_dir, exist_ok=True)
os.makedirs(memes_dir, exist_ok=True)

# Iterate through images in the directory
files = os.listdir(dir)
for file in files:
    # Read image
    img = imread(os.path.join(dir, file))
    
    # Resize image (you might need to resize to match the input shape of your model)
    resized_img = resize(img, (256, 256))
    
    # Normalize image
    normalized_img = resized_img / 255.0
    
    # Perform prediction
    yhat = model.predict(np.expand_dims(normalized_img, axis=0))[0]
    
    # Classify image and move to appropriate directory
    if yhat > 0.5:
        move(os.path.join(dir, file), os.path.join(regular_dir, file))
    else:
        move(os.path.join(dir, file), os.path.join(memes_dir, file))
