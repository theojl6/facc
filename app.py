import os
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import cv2


IMAGE_UPLOADS = 'static/uploads/'

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = IMAGE_UPLOADS

model = pickle.load(open('models/facc_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    if request.files:
        img = request.files["image"]
        img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
    img = cv2.imread(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
    img = cv2.resize(img, (256, 256))
    img = img / 255.
    img = img.reshape(1, 256 * 256 * 3)
    prediction = model.predict(img)
    output = prediction[0]
    return render_template('index.html', prediction_text='Image is {}'.format(output))
    

if __name__ == "__main__":
    app.run(debug=True)

