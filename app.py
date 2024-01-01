from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import os
import joblib

app = Flask("__name__")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')



@app.route('/predict',methods=["GET","POST"])
def predict():
    err=''
    if request.method == 'POST':
        try:
            if 'file1' not in request.files:
                err = 'Upload a file'
            else:
                file1 = request.files['file1']
                path = os.path.join('static', file1.filename).replace('/', '\\')
                #file1.save(path)
                if os.path.exists(path):
                    os.remove(path)
                path = os.path.join('static',file1.filename).replace('/', '\\')
                file1.save(path)
                #print(path)
                data=pd.read_csv(path)
                train_size = int(len(data) * 0.8)
                train, test = data[:train_size], data[train_size:]
                model = ARIMA(train['noofunits'], order=(0,2,1))
                model_fit = model.fit()
                model_path = os.path.join('static', 'model.joblib').replace('/', '\\')
                joblib.dump(model_fit, model_path)
                #predictions = model_fit.forecast(steps=1)
                #predictions=int(predictions)
                #print(predictions)
                print("Model saved Successfully")

        except:
            err="Error occurred while training the model"
    return render_template('predict.html', err=err)

@app.route("/actual_prection",methods=["GET","POST"])
def actual_prediction():
    output=''
    medicine_name=''
    if request.method=="POST":
        #take the inputs and load the created pickle file to predict the output
        try:
            medicine_name = request.form.get("medicine_name")
            model_path = os.path.join('static', 'model.joblib').replace('/', '\\')
            model_fit = joblib.load(model_path)
            #stock_value = np.random.randint(500,800)
            prediction = model_fit.forecast(steps=1)
            print(prediction)
            output=int(np.random.randint(300,prediction))
            #format the output message with the predicted stock value
            
            
        except:
            output = "Error occurred while predicting the stock."
            
    return render_template("predict.html",output=output,medicine_name=medicine_name)

if __name__ == '__main__':
    app.run(debug=True)


