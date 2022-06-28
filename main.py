from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.interpolate import interp1d
from statistics import mode

app = FastAPI()

model = tf.keras.models.load_model("model_payment_history_sampling.h5")
ohe = joblib.load("ohe_payment_history.gz")
sc = joblib.load("sc_payment_history.gz")
sc_std_pd = joblib.load("sc_std_pd_payment_history.gz")
sc_mean_pd = joblib.load("sc_mean_pd_payment_history.gz")

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
        status.append(3 if paymentDate[i] >= 1 and paymentDate[i] < 10 else (2 if paymentDate[i] >= 10 and paymentDate[i] < 20 else 1))

    def convert_LTV(billing_10, billing_9, billing_8, billing_7, billing_6, billing_5, billing_4, billing_3, billing_2, billing_1):
    	if billing_10 != 32:
            return 10
    	elif billing_10 == 32 and billing_9 != 32:
            return 9
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 != 32:
            return 8
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 == 32 and billing_7 != 32:
            return 7
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 == 32 and billing_7 == 32 and billing_6 != 32:
            return 6
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 == 32 and billing_7 == 32 and billing_6 == 32 and billing_5 != 32:
            return 5
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 == 32 and billing_7 == 32 and billing_6 == 32 and billing_5 == 32 and billing_4 != 32:
            return 4
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 == 32 and billing_7 == 32 and billing_6 == 32 and billing_5 == 32 and billing_4 == 32 and billing_3 != 32:
            return 3
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 == 32 and billing_7 == 32 and billing_6 == 32 and billing_5 == 32 and billing_4 == 32 and billing_3 == 32 and billing_2 != 32:
            return 2
    	elif billing_10 == 32 and billing_9 == 32 and billing_8 == 32 and billing_7 == 32 and billing_6 == 32 and billing_5 == 32 and billing_4 == 32 and billing_3 == 32 and billing_2 == 32 and billing_1 != 32:
            return 1
    	else:
            return 0

    ltv = convert_LTV(paymentDate[0], paymentDate[1], paymentDate[2], paymentDate[3], paymentDate[4], paymentDate[5], paymentDate[6], paymentDate[7], paymentDate[8], paymentDate[9])

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

    if ltv < 10 and ltv > 1:
        x_temp = np.linspace(1, int(ltv), num = int(ltv), endpoint = True)
        xnew = np.linspace(1, int(ltv), num = 10, endpoint = True)

        #paymentDate
        y_temp = data[0, :, 0][-int(ltv):]
        f = interp1d(x_temp, y_temp, kind = "linear")
        data[0, :, 0] = np.round(f(xnew))

    elif ltv == 1:
        data[0, :, 0] = np.array([data[0, -1, 0] for j in range(10)])

    std_pd = np.std(np.array([paymentDate[0],
                              paymentDate[1],
                              paymentDate[2],
                              paymentDate[3],
                              paymentDate[4],
                              paymentDate[5],
                              paymentDate[6],
                              paymentDate[7],
                              paymentDate[8],
                              paymentDate[9]]))
    mean_pd = np.mean(np.array([paymentDate[0],
                                paymentDate[1],
                                paymentDate[2],
                                paymentDate[3],
                                paymentDate[4],
                                paymentDate[5],
                                paymentDate[6],
                                paymentDate[7],
                                paymentDate[8],
                                paymentDate[9]]))
    std_stat = np.std(np.array([status[0],
		                status[1],
                   	        status[2],
                      	        status[3],
                   	        status[4],
                  	        status[5],
                  	        status[6],
                  	        status[7],
                	        status[8],
               		        status[9]]))
    mean_stat = np.mean(np.array([status[0],
		                  status[1],
                   	          status[2],
                      	          status[3],
                   	          status[4],
                  	          status[5],
                  	          status[6],
                  	          status[7],
                	          status[8],
               		          status[9]]))
    print(std_pd, mean_pd, std_stat, mean_stat)
    for i in range(data.shape[0]):
    	data[i, :] = sc.transform(data[i, :])
    
    std_pd_standardize = sc_std_pd.transform([[std_pd]]).reshape(-1)[0]
    mean_pd_standardize = sc_mean_pd.transform([[mean_pd]]).reshape(-1)[0]
    print(std_pd_standardize, mean_pd_standardize)

    data2 = np.array([[std_pd_standardize, mean_pd_standardize]])

    result = model.predict([data, data2])

    if paymentDate[9] == 32:
        status = "Telat Bayar"
        pred_late_paid = float(100)
        pred_paid = float(50)
        pred_loyal_paid = float(50)
    elif std_pd > 4.36:
        if np.argmax(result, axis = 1) == 0:
            status = "Telat Bayar"
        elif np.argmax(result, axis = 1) == 1:
            status = "Bayar Tepat Waktu"
        else:
            status = "Rajin Bayar"
        pred_late_paid = np.round(float(result[0][0]) * 100, 5)
        pred_paid = np.round(float(result[0][1]) * 100, 5)
        pred_loyal_paid = np.round(float(result[0][2]) * 100, 5)
    else:
        if np.round(mean_stat) == 1:
            status = "Telat Bayar"
            pred_late_paid = ((15 - std_pd) / 15) * 100
            pred_paid = 100 - (pred_late_paid / 2)
            pred_loyal_paid = 100 - (pred_late_paid *2 / 3)
        elif np.round(mean_stat) == 2:
            status = "Bayar Tepat Waktu"
            pred_paid = ((15 - std_pd) / 15) * 100
            pred_late_paid = 100 - (pred_paid / 3)
            pred_loyal_paid = 100 - (pred_paid / 3)
        else:
            status = "Rajin Bayar"
            pred_loyal_paid = ((15 - std_pd) / 15) * 100
            pred_paid = 100 - (pred_loyal_paid / 2)
            pred_late_paid = 100 - (pred_loyal_paid * 2 / 3)

    print(result)
    print(status)
    paymentDate = np.array(paymentDate)

    try: 
        modus = mode(paymentDate)
    except: 
        modus = "-"

    return {
        "Persentase Prediksi Telat Bayar": np.round(pred_late_paid, 5),
	"Persentase Prediksi Bayar Tepat Waktu": np.round(pred_paid, 5),
	"Persentase Prediksi Rajin Bayar": np.round(pred_loyal_paid, 5),
        "Deksripsi Prediksi": status,
	"Tanggal Bayar Tertinggi": int(np.max(paymentDate[paymentDate != 32])),
	"Tanggal Bayar Terendah": int(np.min(paymentDate)),
	"Standar Deviasi Tanggal Pembayaran": np.round(float(std_pd), 5),
        "Standar Deviasi Status (Segmentasi Tanggal Pembayaran)": np.round(float(std_stat), 5),
        "Rata-Rata Tanggal Pembayaran": np.round(mean_pd, 5),
        "Modus Tanggal Pembayaran": int(modus)
    }