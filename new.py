import os
import numpy as np
from PIL import Image
import tensorflow as tf
from werkzeug.utils import secure_filename
from utils import load_model, preprocess_image, postprocess_image
from flask import Flask, flash, request, redirect, url_for

PATH = 'recolor.h5'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

appy = Flask(__name__, template_folder='pages')
appy.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@appy.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = os.path.join(appy.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            # return redirect(url_for('download_file', name=filename))
            img = run_inference(filename)
            img.save('/Users/adityashukla/Desktop')
            return "Done!"
    
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def run_inference(img_path):
    model = load_model(PATH)
    
    input_img = Image.open(img_path)
    input_img = preprocess_image(input_img)
    output_img = model.predict(input_img, verbose = 0)
    colorized_img = postprocess_image(output_img)
    
    return colorized_img

if __name__ == "__main__":
    appy.run(debug=True)
