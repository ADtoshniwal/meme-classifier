

from tensorflow.keras.models import load_model
from cv2 import imread
from matplotlib.pyplot import imshow
import os
from tensorflow.image import resize
from numpy import expand_dims
from shutil import move

# Load model
model = load_model('imageclassifier_44mb.h5')

# 1) load images
# 2) resize them
# 3) cal yhat
# 4) if yhat is > 0.5 -> regular else meme


dir = 'data'
files = os.listdir(dir)
for file in files:
    img = imread(os.path.join(dir, file))
    res = resize(img, (256, 256))
    yhat = model.predict(expand_dims(res/255, 0), verbose=0)
    if yhat > 0.5:
        move(os.path.join(dir, file), os.path.join('regular', file))
    else:
        move(os.path.join(dir, file), os.path.join('memes', file))
