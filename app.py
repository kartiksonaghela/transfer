from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)
model = tf.keras.models.load_model("model_resnet50.h5")



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
   

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    dict1={0:'phlox',1:'rose',2:'calendula',3:'iris',4:'leucanthemum maximum',
                 5:'bellflower',6:'viola',7:'rudbeckia laciniata',
                 8:'peony',9:'aquilegia'}
    a=preds[0]
    a=dict1[a]
   
    return a


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=[ 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
    return render_template('index.html', prediction_text="The name of the flower is {}".format(result))

if __name__ == '__main__':
    app.run(debug=True)
