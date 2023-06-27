# Transportation Stock Analysis
* Instructor :  <b>Dinar Ajeng Kristiyanti, S.Kom., M.Kom.</b>
* Place : <b>Universitas Multimedia Nusantara</b>
* Course : <b>Bachelor of Computer Science Research</b>

## Table of Contents
* [Background](#background)
* [Requirement](#requirement)
* [Inspiration](#inspiration)
* [Schema](#schema)

## Background
<img src="https://github.com/Bayunova28/Transportation_Stock_Analysis/blob/master/background.jpg" height="450" width="1100">

<p align="justify">Inflation growth in Indonesia and other countries impacts the currency value and investors' purchasing power, particularly in the transportation sector. Early prediction efforts using data mining techniques like Long Short-Term Memory (LSTM) with different activation and optimizer
parameters can help investors make informed decisions and prevent losses. This research implemented LSTM with various activation parameters (linear, relu, sigmoid, and tanh) and optimizers
(adam, adagrad, nadam, rmsprop, adadelta, SGD, and adamax) using the CRISP-DM framework,
Python programming language, and Visual Studio Code tools. The aim was to predict stock
prices in Indonesia's transportation sector. The results showed promising evaluation metrics
(MAE, MAPE, MSE, R-Squared, RMSE) and statistical tests (Shapiro-Wilk) for Air Asia Indonesia's stock (CMPP.JK) within an elapsed time of 82.16 minutes. Furthermore, an agile software
development approach was used to develop a web-based information system using streamlit and
the LSTM model.</p>

## Requirement
* **Python 3.11.2**
* **Tensorflow 2.12.0**
* **Visual Studio Code 1.77.0**
* **Streamlit 1.19.0**

## Inspiration
* Evaluate the success of various activation and optimizer comparisons used to measure the performance of MAPE, MAE, RMSE, MSE, R-squared, elapsed time, and statistical tests.
* Predict transportation stock prices using the Long Short-Term Memory (LSTM) algorithm model with various activation comparisons such as linear, relu, tanh, and sigmoid, as well as optimizers such as adam, adagrad, nadam, rmsprop, adadelta, SGD, and adamax.
* Build Long Short-Term Memory (LSTM) algorithm model using activation and optimizer comparisons to predict transportation stock prices in a web-based information system.

## Schema
* **Date :** period of time the stock move
* **High :** high price of stock 
* **Low :** low price of stock
* **Open :** open price of stock
* **Close :** close price of stock
* **Volume :** volume of stock

## Datasets
| Ticker  | Stock Name | Source  | 
| ------- | ---------- | ------- |
| AKSI.JK | Mineral Sumberdaya Mandiri Tbk. | https://finance.yahoo.com/quote/AKSI.JK/history?p=AKSI.JK |
| CMPP.JK | Air Asia Indonesia Tbk. | https://finance.yahoo.com/quote/CMPP.JK/history?p=CMPP.JK |
| SAFE.JK | Steady Safe Tbk. | https://finance.yahoo.com/quote/SAFE.JK/history?p=SAFE.JK |
| SMDR.JK | Samudera Indonesia Tbk. | https://finance.yahoo.com/quote/SMDR.JK/history?p=SMDR.JK |
| TMAS.JK | Temas Tbk. | https://finance.yahoo.com/quote/TMAS.JK/history?p=TMAS.JK |
| WEHA.JK | WEHA Transportasi Indonesia Tbk. | https://finance.yahoo.com/quote/WEHA.JK/history?p=WEHA.JK |
  
## Acknowledgement
The authors would like to thank Universitas Multimedia Nusantara for providing support in this research.
