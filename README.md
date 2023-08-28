# Transportation Stock Analysis
* Instructor :  <b>Dinar Ajeng Kristiyanti, S.Kom., M.Kom.</b>
* Place : <b>Universitas Multimedia Nusantara</b>
* Course : <b>Computer Science Research</b>

## Table of Contents
* [Background](#background)
* [Requirement](#requirement)
* [Inspiration](#inspiration)
* [Schema](#schema)

## Background
<img src="https://github.com/Bayunova28/Transportation_Stock_Analysis/blob/master/background.jpg" height="450" width="1100">

<p align="justify">Inflation growth occurs in several international countries as well as Indonesia. This causes currency values to rise and affects the purchasing power of stocks for investors. Stocks affected by inflation growth occur in various sectors, especially transportation. This requires early prediction efforts to be made so that investors can make decisions in making investments so that no losses occur. One solution that will be carried out is to use data mining techniques using a deep learning approach, namely Long Short-Term Memory (LSTM) using activation and optimizer parameter comparisons. The activation parameters used include linear, relu, sigmoid, and tanh. While the optimizers used include adam, adagrad, nadam, rmsprop, adadelta, SGD, and adamax with elapsed time and statistical tests. This research uses the CRISP-DM framework with the Python programming language and the help of tools from Visual Studio Code and aims to predict transportation stock prices in Indonesia.</p>

## Requirement
* **Python 3.11.2**
* **Tensorflow 2.12.0**
* **Visual Studio Code 1.77.0**
* **Streamlit 1.19.0**
* **Jupyter Notebook / Google Colaboratory**
* **Keras**
* **Pillow**
* **Scikit-Learn**
* **Datetime**
* **Plotly**
* **Numpy**
* **Openpyxl**
* **Python Math**

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
| WEHA.JK | WEHA Transportasi Indonesia Tbk.    | https://finance.yahoo.com/quote/WEHA.JK/history?p=WEHA.JK |
  
## Acknowledgement
The authors would like to thank Multimedia Nusantara University for providing support in this research.
