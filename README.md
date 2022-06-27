# Payment History Analysis to Predict Payment Status Using LSTM (Single)

## Data
Payment History from Telkom DDB, feature used in the model:
* Billing Amount Total
* Billing Payment Date
* Billing Period

## Team Member
1. Syachrul Qolbi Nur Septi

## Table of Contents
1. [Requirements](#requirements) to install on your system
2. [Results](#results)
3. [Links to google colab]([https://colab.research.google.com/drive/17Ews_Ol0RjeU69ewKKElYorFSuWymRtb?usp=sharing](https://colab.research.google.com/drive/1C2XGCS-81jo9YKU4oo-yN2W5zJsxdH45?usp=sharing))
4. [Tutorial](#tutorial)

## Requirements

The main requirements are listed below:

Tested with 
* Tensorflow 2.7.0
* Python 3.7.10
* Numpy 1.19.5
* Matplotlib 3.2.2
* Pandas 1.1.5
* Scipy 1.8.0

Additional requirements to generate dataset:

* Os
* Sklearn.metrics import classification_report, confusion_matrix
* Sklearn.preprocessing import StandardScaler
* Shutil
* Google.colab import drive
* FastAPI
* Joblib
* Pickle


## Results
These are the results for our models.

### LSTM Model (Unsubscribe/Subscribe)
<div class="tg-wrap"><table class="tg">
  <tr>
    <th class="tg-7btt" colspan="6">Result</th>
  </tr>
  <tr>
    <td class="tg-7btt"></td>
    <td class="tg-7btt">Accuracy (Macro Avg)</td>
    <td class="tg-7btt">Precision</td>
    <td class="tg-7btt">Recall</td>
    <td class="tg-7btt">F1-Score</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Late Paid</td>
    <td class="tg-c3ow">70.48%</td>
    <td class="tg-c3ow">78.07%</td>
    <td class="tg-c3ow">65.96%</td>
    <td class="tg-c3ow">71.51%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Paid</td>
    <td class="tg-c3ow">70.48%</td>
    <td class="tg-c3ow">56.44%</td>
    <td class="tg-c3ow">77.79%</td>
    <td class="tg-c3ow">65.42%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Loyal Paid</td>
    <td class="tg-c3ow">70.48%</td>
    <td class="tg-c3ow">85.19%</td>
    <td class="tg-c3ow">66.19%</td>
    <td class="tg-c3ow">74.50%</td>
  </tr>
</table></div>

## Tutorial
# Instalasi Python

Pastikan sudah terinstall python dan pip dalam system anda, jika system anda mengunakan linux bisa mengikuti command di bawah ini

`
sudo apt install python3-pip
`

jika system mengunakan OS windows atau yang lain bisa minjau situs resmi python untuk instalasi python https://www.python.org/downloads/

# Instalasi Dependency 
Agar code dapat berjalan di perlukan beberapa dependecy, dapat langsung menjalankan command di terminal berikut satu demi satu jika python dan pip sudah terinstall

```
pip install fastapi
pip install uvicorn
pip install tensorflow
```

# Menjalankan API
Untuk menjalankan API cukup mejalankan command berikut di terminal
```
uvicorn main:app --reload
```
Secara default dia akan jalan secara lokal di 127.0.0.1 dengan port 8000 

Output jika runnning berhasil

![image](/Images/Output_Uvicorn.png) 

Jika ingin di jalankan di port dan address host yang berbeda bisa mengunakan option --host dan --port
```
uvicorn --host [host address] --port [nilai port]  main:app --reload 
```

# Menggunakan API
Kita akan memprediksi status berlangganan user, apakah user tersebut akan berlangganan kembali di bulan depan atau tidak. Untuk mengunakan endpoint bisa dengan menyiapkan body parameternya berupa format JSON dengan format seperti berikut

```
{
    "billing_11_amountTotal": "38.497,00",
    "billing_10_amountTotal": "144.950,00",
    "billing_9_amountTotal": "148.497,00",
    "billing_8_amountTotal": "148.497,00",
    "billing_7_amountTotal": "147.765,00",
    "billing_6_amountTotal": "38.497,00",
    "billing_5_amountTotal": "51.804,00",
    "billing_4_amountTotal": "50.482,00",
    "billing_3_amountTotal": "49.947,00",
    "billing_2_amountTotal": "49.947,00",
    "billing_1_amountTotal": "49.947,00",

    "billing_11_paymentDate": "20210819",
    "billing_10_paymentDate": "20210904",
    "billing_9_paymentDate": "20211003",
    "billing_8_paymentDate": "20211108",
    "billing_7_paymentDate": "20211205",
    "billing_6_paymentDate": "20220107",
    "billing_5_paymentDate": "20220202",
    "billing_4_paymentDate": "20220310",
    "billing_3_paymentDate": "20220402",
    "billing_2_paymentDate": "20220503",
    "billing_1_paymentDate": "20220609",

    "billing_11_period": "202108",
    "billing_10_period": "202109",
    "billing_9_period": "202110",
    "billing_8_period": "202111",
    "billing_7_period": "202112",
    "billing_6_period": "202201",
    "billing_5_period": "202202",
    "billing_4_period": "202203",
    "billing_3_period": "202204",
    "billing_2_period": "202205",
    "billing_1_period": "202206"
}
```
dan untuk URL API mengunakan format sebagai berikut
```
http://[Host]:[Port]/predict
```
dan request method yang digunakan adalah **GET** 
API akan mengembalikan variabel Percentage dan Predict Description beserta valuenya dengan tipe data JSON.

## Hasil Retun API
```
{
    "Persentase Prediksi Telat Bayar": 20.11317,
    "Persentase Prediksi Bayar Tepat Waktu": 31.52434,
    "Persentase Prediksi Rajin Bayar": 48.36249,
    "Deksripsi Prediksi": "Rajin Bayar",
    "Tanggal Bayar Tertinggi": 19,
    "Tanggal Bayar Terendah": 2,
    "Standar Deviasi Tanggal Pembayaran": 4.94065,
    "Standar Deviasi Status (Segmentasi Tanggal Pembayaran)": 0.4
}
```
## Contoh mengunakan POSTMAN
![image](/Images/Contoh_Postman.png)
