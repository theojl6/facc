import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import matplotlib.image as mpimg
import os
import cv2
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT-PATH'] = 5000000

model = pickle.load(open("models/svc_facc.pickle", "rb"))

def plot_category()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    print(request.files)
    if 'image' not in request.files:
        print('File not uploaded')
        return
    img = request.files['image']
    img = mpimg.imread(img)

    img = cv2.resize(img, (256, 256))
    img = img / 255.
    img = img.reshape(1, 256 * 256 * 3)
    prediction = model.predict(img)
    output = prediction[0]
    result = ('fac' if output == 1 else 'not fac')
    return render_template('index.html', prediction_text='Image is {}, with probability {}'.format(result, output))
    

if __name__ == "__main__":
    app.run(debug=True)