import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.image as mpimg

IMAGE_UPLOADS = 'static/uploads/'

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = IMAGE_UPLOADS

model = keras.models.load_model('models/fine_tuned_xception_facc')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    if request.files:
        img = request.files["image"]
#        img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
    img = mpimg.imread(img) # uses mpimg.imread() instead of cv2.imread() because of type error
    img = cv2.resize(img, (256, 256))
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    output = prediction[0][0]
    result = ('fac' if output >= 0 else 'not fac')
    return render_template('index.html', prediction_text='Image is {} with probability {}'.format(result, output))
    

if __name__ == "__main__":
    app.run(debug=True)

