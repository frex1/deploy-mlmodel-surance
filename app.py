from flask import Flask, request, url_for, redirect,render_template, jsonify
from pycaret.regression import *
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
model = load_model('insurance_deploy')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
	return render_template("index.html")
@app.route('/predict', methods = ['POST'])
def predict():
	int_features = [x for x in request.form.values()]
	final = np.array(int_features)
	data_unseen = pd.DataFrame([final], columns = cols)
	predicion = predict_model(model, data = data_unseen, round = 0)
	prediction = int(prediction.Label[0])
	return render_template('index.html', pred='Expected Bill will be {}'.format(prediction))