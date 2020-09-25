# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename='cement_model.pkl'
with open(filename, 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
        cement = int(request.form['cement'])
        blast = int(request.form['blast_furnace_slag'])
        fly = int(request.form['fly_ash'])
        wat = int(request.form['water'])
        superi = int(request.form['superplasticizer'])
        coarse = int(request.form['coarse_aggregate'])
        fine_agg = int(request.form['fine_aggregate'])
        age = int(request.form['age'])
        
        data = np.array([[cement,blast,fly,wat,superi,coarse,fine_agg,age]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
    

if __name__ == '__main__':
	app.run(debug=True)