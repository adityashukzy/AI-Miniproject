import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
from utils import load_model, preprocess_image, postprocess_image
from flask import Flask, render_template, request

PATH = 'recolor.h5'
UPLOAD_FOLDER = 'uploads'

app = Flask(__name__, template_folder='pages')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods = ['GET', 'POST'])
def upload_file():
    return render_template('upload.html')

@app.route("/uploader", methods = ['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)
        img = Image(file_path)

        colorized_img = run_inference(img)
        colorized_img = Image.fromarray(colorized_img)

        colorized_img.save('/Users/adityashukla/Desktop')

        return "Image uploaded!"

def run_inference(input_img):
    model = load_model(PATH)
    
    input_img = preprocess_image(input_img)
    output_img = model.predict(input_img, verbose = 0)
    colorized_img = postprocess_image(output_img)
    
    return colorized_img

if __name__ == "__main__":
    app.run(debug=True)
