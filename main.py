from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.interpolate import interp1d

app = FastAPI()

model = tf.keras.models.load_model("model_payment_history_sampling.h5")
ohe = joblib.load("ohe_payment_history.gz")
sc = joblib.load("sc_payment_history.gz")

class Data(BaseModel):
    billing_11_amountTotal: str
    billing_10_amountTotal: str
    billing_9_amountTotal: str
    billing_8_amountTotal: str
    billing_7_amountTotal: str
    billing_6_amountTotal: str
    billing_5_amountTotal: str
    billing_4_amountTotal: str
    billing_3_amountTotal: str
    billing_2_amountTotal: str
    billing_1_amountTotal: str

    billing_11_paymentDate: str
    billing_10_paymentDate: str
    billing_9_paymentDate: str
    billing_8_paymentDate: str
    billing_7_paymentDate: str
    billing_6_paymentDate: str
    billing_5_paymentDate: str
    billing_4_paymentDate: str
    billing_3_paymentDate: str
    billing_2_paymentDate: str
    billing_1_paymentDate: str

    billing_11_period: str
    billing_10_period: str
    billing_9_period: str
    billing_8_period: str
    billing_7_period: str
    billing_6_period: str
    billing_5_period: str
    billing_4_period: str
    billing_3_period: str
    billing_2_period: str
    billing_1_period: str

@app.get("/predict")
def predict(data: Data):
    paymentDate = [data.billing_11_paymentDate,
		   data.billing_10_paymentDate,
                   data.billing_9_paymentDate,
                   data.billing_8_paymentDate,
                   data.billing_7_paymentDate,
                   data.billing_6_paymentDate,
                   data.billing_5_paymentDate,
                   data.billing_4_paymentDate,
                   data.billing_3_paymentDate,
                   data.billing_2_paymentDate]
    period = [data.billing_11_period,
   	      data.billing_10_period,
	      data.billing_9_period,
              data.billing_8_period,
              data.billing_7_period,
              data.billing_6_period,
              data.billing_5_period,
              data.billing_4_period,
              data.billing_3_period,
              data.billing_2_period]
    status = []

    for i in range(10):
        paymentDate[i] = int(paymentDate[i][-2:]) if int(paymentDate[i][-4:-2]) == int(period[i][-2:]) and int(paymentDate[i][-2:]) != 0 else (int(32) if int(paymentDate[i][-2:]) == 0 else int(31))
        status.append(2 if paymentDate[i] >= 1 and paymentDate[i] < 10 else (1 if paymentDate[i] >= 10 and paymentDate[i] < 20 else 0))

    data = [[paymentDate[0]],
            [paymentDate[1]],
            [paymentDate[2]],
            [paymentDate[3]],
            [paymentDate[4]],
            [paymentDate[5]],
            [paymentDate[6]],
            [paymentDate[7]],
            [paymentDate[8]],
            [paymentDate[9]]]
            
    data = np.array([data])
    print(data)
    std = np.std(np.array([[data[0, 0][0]],
		           [data[0, 1][0]],
                   	   [data[0, 2][0]],
                  	   [data[0, 3][0]],
                   	   [data[0, 4][0]],
                  	   [data[0, 5][0]],
                  	   [data[0, 6][0]],
                  	   [data[0, 7][0]],
                	   [data[0, 8][0]],
               		   [data[0, 9][0]]]))

    for i in range(data.shape[0]):
    	data[i, :] = sc.transform(data[i, :])

    result = model.predict([data])

    if np.argmax(result, axis = 1) == 0:
        status = "Late Paid"
    elif np.argmax(result, axis = 1) == 1:
        status = "Paid"
    else:
        status = "Loyal Paid"
    print(result)
    print(status)
    return {
        "Percentage Late Paid": np.round(float(result[0][0]), 2),
	"Percentage Paid": np.round(float(result[0][1]), 2),
	"Percentage Loyal Paid": np.round(float(result[0][2]), 2),
        "Predict Description": status,
	"High Payment Date": int(np.max(paymentDate)),
	"Low Payment Date": int(np.min(paymentDate)),
	"Standard Deviation Payment Date": np.round(float(std), 2),
    }