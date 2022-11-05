import requests

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

import pandas as pd

import tensorflow as tf

from flask import Flask, request, render_template, redirect, url_for, app
import os

from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction')
def prediction():
    return render_template('predict.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(filepath)
        img = image.load_img(filepath, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        plant = request.form['plant']
        print(plant)

        if(plant == 'vegetable'):
            model = load_model("vegetable.h5")
            preds = model.predict_classes(x)
            print(preds)
            df = pd.read_excel('precautions_veg.xlsx')
            print(df.iloc[preds[0]]['caution'])
        
        else:
            model = load_model("vegetable.h5")
            preds = model.predict_classes(x)
            df = pd.read_excel('precautions_fruit.xlsx')
            print(df.iloc[preds[0]]['caution'])

    return df.iloc[preds[0]]['caution']


if __name__ == "__main__":
    app.run(debug=False)
    
