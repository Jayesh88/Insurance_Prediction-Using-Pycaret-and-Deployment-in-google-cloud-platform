# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 22:37:27 2020

@author: HP PC
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
from pycaret.regression import *
import pickle
import pandas as pd

app = Flask(__name__)
model = load_model('store.pickle.pkl')
#model = pickle.load(open('store.pickle.pkl', 'rb'))
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template('Jayesh_abc.html')

    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    data_unseen = pd.DataFrame([final_features], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])


    return render_template('store.pickle.pkl', prediction_text='Predicted Insurance price $ {}'.format(prediction))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)