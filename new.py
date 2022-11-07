from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/cnn_model.h5'
# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()     

# F:\Deployment-Deep-Learning-Model-master\new.py
def model_predict(img_path, model):
    print("hello")
    data = []
    x_test = []
    try:
        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (150, 150))
    except Exception as e:
                print(e)
    print("working til here")
    x_test.append(resized_arr)
    x_test = np.array(x_test) / 255
    x_test = x_test.reshape(-1, 150, 150, 1)
    # predictions = model.predict_classes(x_test)
    predictions = (model.predict(x_test) > 0.5).astype("int32")
    predictions = predictions.reshape(1,-1)[0]
    print(predictions)
    # # preds = model.predict(resized_arr)
    if(predictions[0] == 0):
        return "The person has Pneumonia"
    return "The person is normal"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("uploaded")
        # Make prediction
        preds = model_predict(file_path, model)
        result =  preds
        print(type(result))
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
