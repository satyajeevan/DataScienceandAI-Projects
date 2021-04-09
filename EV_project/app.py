from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import os
import sys

app = Flask(__name__)
model = pickle.load(open('Ev_RFmodel.pkl','rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('indexs.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        NV = float(request.form['NV'])
        Current = float(request.form['Current'])
        OverWeight = int(request.form['OverWeight'])
        CRating = float(request.form['CRating'])
        AgeOfBattery = int(request.form['AgeOfBattery'])
        OutsideTemp = int(request.form['OutsideTemp'])
        AuxLoad = int(request.form['AuxLoad'])
        ReducedRange = int(request.form['ReducedRange'])
        WeeklyChargeCycles = int(request.form['WeeklyChargeCycles'])
        ChargingType = request.form['ChargingType']
        DrivingCond = request.form['DrivingCond']

        to_predict_list = [DrivingCond,ChargingType,WeeklyChargeCycles,AuxLoad,NV, Current, OverWeight, CRating, AgeOfBattery, OutsideTemp,
                           ReducedRange]
        
        final=np.array(to_predict_list).reshape(1,11)
        res = model.predict(final)
        print(res[0])
        
        if res[0]==1:
            prediction='Yes'
        else:
            prediction='No'
        return render_template("indexs.html", prediction_text=prediction)

    else:
        return render_template('indexes.html')


if __name__ == "__main__":
    app.run(debug=False)


