import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

def load_model(path):
    model = keras.models.load_model(path)
    print(model.summary())
    return model

def preprocess_image(img):
    img = np.array(img)
    normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
    img = normalization_layer(img)
    img = tf.image.resize(img, [1024,1024])
    print(img.shape)
    img = tf.image.rgb_to_hsv(img)
    return img[:, :, :, -1:]

def postprocess_image(img):
    img = tf.image.hsv_to_rgb(img)
    img = np.array(img)
    return Image.fromarray(img)

def run_inference(model, input_img):
    input_img = preprocess_image(input_img)
    output_img = model.predict(input_img, verbose = 0)
    colorized_img = postprocess_image(output_img)
    
    return colorized_img
