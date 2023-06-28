# Import library
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
from tensorflow import keras
from keras.models import load_model
import datetime
import math
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import numpy as np
from sklearn import metrics

# Set web page layouts
st.set_page_config(
    page_title='Transportation Stock Exchange',
    page_icon='assets/transport-icon.png',
    layout='wide'
)

# Set web page icon
img = Image.open('assets/transport-icon.png')
st.sidebar.image(img, use_column_width='auto')

# Adding sidebar header
st.sidebar.write('# Transportation Stock Exchange')

# Adding ticker selection
ticker = st.sidebar.selectbox('Choose Your Ticker',
                              ('AKSI', 'CMPP', 'SAFE', 'SMDR', 'TMAS', 'WEHA'))

# Adding LSTM activation selection
parameter = st.sidebar.selectbox('Parameter (Activation - Optimizer)',
                                  ('Default', 'Linear - Adam', 'Linear - AdaGrad',
                                   'Linear - Nadam', 'Linear - RMSProp', 'Linear - AdaDelta',
                                   'Linear - SGD', 'Linear - AdaMax', 'ReLU - Adam', 'ReLU - AdaGrad',
                                   'ReLU - Nadam', 'ReLU - RMSProp', 'ReLU - AdaDelta', 'ReLU - SGD',
                                   'ReLU - AdaMax', 'Sigmoid - Adam', 'Sigmoid - AdaGrad',
                                   'Sigmoid - Nadam', 'Sigmoid - RMSProp', 'Sigmoid - AdaDelta',
                                   'Sigmoid - SGD', 'Sigmoid - AdaMax', 'Tanh - Adam', 'Tanh - AdaGrad',
                                   'Tanh - Nadam', 'Tanh - RMSProp', 'Tanh - AdaDelta', 'Tanh - SGD',
                                   'Tanh - AdaMax'))

# Define the filter range
filter_start_date = datetime.datetime(2023, 4, 1)
filter_end_date = datetime.datetime(2023, 10, 1)

# Display the date inputs with validation
start_date = st.sidebar.date_input('Start Date', filter_start_date, min_value=filter_start_date, max_value=filter_end_date)
end_date = st.sidebar.date_input('End Date', filter_end_date, min_value=filter_start_date, max_value=filter_end_date)

# Display stock analysis
################################################# AKSI.JK ###############################################
if ticker == 'AKSI':
    st.image(Image.open('assets/aksi-icon.png'), use_column_width='auto')
    st.write('#### Dataset')

    # Read the data
    df_aksi = pd.read_excel(
        'transportation_stocks.xlsx', sheet_name='AKSI.JK')

    # Convert the date column to datetime format
    df_aksi['Date'] = [datetime.datetime.strptime(str(target_date).split(
        ' ')[0], '%Y-%m-%d').date() for target_date in df_aksi['Date']]

    # Fill missing value using mean imputation
    for i in df_aksi[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]:
        df_aksi[i].fillna(df_aksi[i].mean(), inplace=True)

    # Display the dataframe with font size of 14
    st.checkbox('Use container width', value=True, key='use_container_width')
    st.dataframe(
        df_aksi, use_container_width=st.session_state.use_container_width)

    st.write(' #### Stock Prices History')
    fig_att = go.Figure()
    fig_att.add_trace(go.Scatter(x=df_aksi['Date'], y=df_aksi['Open'], mode='lines',
                                 name='Open', line=dict(color='#1f77b4', width=3)))
    fig_att.add_trace(go.Scatter(x=df_aksi['Date'], y=df_aksi['High'], mode='lines',
                                 name='High', line=dict(color='#ff7f0e', width=3)))
    fig_att.add_trace(go.Scatter(x=df_aksi['Date'], y=df_aksi['Low'], mode='lines',
                                 name='Low', line=dict(color='#2ca02c', width=3)))
    fig_att.add_trace(go.Scatter(x=df_aksi['Date'], y=df_aksi['Close'], mode='lines',
                                 name='Close', line=dict(color='#d62728', width=3)))
    fig_att.add_trace(go.Scatter(x=df_aksi['Date'], y=df_aksi['Adj Close'], mode='lines',
                                 name='Adj Close', line=dict(color='#9467bd', width=3)))
    fig_att.add_trace(go.Scatter(x=df_aksi['Date'], y=df_aksi['Volume'], mode='lines',
                                 name='Volume', line=dict(color='#8c564b', width=3)))
    fig_att.update_layout(
        showlegend=True,
        legend={
            'font': {'size': 15}
        },
        xaxis={
            'rangeslider': {'visible': False},
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'showgrid': True,
            'tickfont': {'size': 15},
            'title': {'text': 'Date', 'font': {'size': 20}}
        },
        yaxis={
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'tickfont': {'size': 15},
            'title': {'text': 'Stock Price (in Rupiah)', 'font': {'size': 20}}
        },
        margin={'t': 25, 'b': 0},
        height=500,
        width=500
    )
    st.plotly_chart(
        fig_att, use_container_width=st.session_state.use_container_width)

    # Select the close price column as the target variable
    target_col = 'Close'
    # Create a new dataframe with only the target variable
    target_df = pd.DataFrame(df_aksi[target_col])
    # Split the data into training and testing sets
    train_size = int(len(target_df) * 0.9)
    train_df = target_df[:train_size]
    test_df = target_df[train_size:]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Define the sequence length and number of features
    sequence_length = 60
    num_features = 1
    # Define the function to create sequences

    def create_sequences(X, y, seq_length):
        X_sequences, y_sequences = [], []
        for i in range(seq_length, len(X)):
            X_sequences.append(X[i-seq_length:i, :])
            y_sequences.append(y[i, :])
        return np.array(X_sequences), np.array(y_sequences)
    # Create training sequences and labels
    X_train, y_train = create_sequences(
        train_scaled, train_scaled, sequence_length)
    # Create testing sequences and labels
    X_test, y_test = create_sequences(
        test_scaled, test_scaled, sequence_length)
    
    if parameter == 'Default':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_lstm_default.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_linear_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_linear_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True) 
    elif parameter == 'Linear - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_linear_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_linear_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_linear_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_linear_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_linear_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)     
    elif parameter == 'ReLU - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_relu_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_relu_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_relu_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_relu_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_relu_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_relu_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_relu_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_sigmoid_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_sigmoid_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_sigmoid_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_sigmoid_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_sigmoid_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_sigmoid_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_sigmoid_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_tanh_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_tanh_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_tanh_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_tanh_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_tanh_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_tanh_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    else:
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/aksi_tanh_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_aksi.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    st.sidebar.write('Parameter Recommendation: ReLU - Nadam')
################################################# CMPP.JK ###############################################
elif ticker == 'CMPP':
    st.image(Image.open('assets/air-asia-icon.png'), use_column_width=False, width=280)
    st.write('#### Dataset')

    # Read the data
    df_cmpp = pd.read_excel(
        'transportation_stocks.xlsx', sheet_name='CMPP.JK')

    # Convert the date column to datetime format
    df_cmpp['Date'] = [datetime.datetime.strptime(str(target_date).split(
        ' ')[0], '%Y-%m-%d').date() for target_date in df_cmpp['Date']]

    # Fill missing value using mean imputation
    for i in df_cmpp[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]:
        df_cmpp[i].fillna(df_cmpp[i].mean(), inplace=True)

    # Display the dataframe with font size of 14
    st.checkbox('Use container width', value=True, key='use_container_width')
    st.dataframe(
        df_cmpp, use_container_width=st.session_state.use_container_width)

    st.write(' #### Stock Prices History')
    fig_att = go.Figure()
    fig_att.add_trace(go.Scatter(x=df_cmpp['Date'], y=df_cmpp['Open'], mode='lines',
                                 name='Open', line=dict(color='#1f77b4', width=3)))
    fig_att.add_trace(go.Scatter(x=df_cmpp['Date'], y=df_cmpp['High'], mode='lines',
                                 name='High', line=dict(color='#ff7f0e', width=3)))
    fig_att.add_trace(go.Scatter(x=df_cmpp['Date'], y=df_cmpp['Low'], mode='lines',
                                 name='Low', line=dict(color='#2ca02c', width=3)))
    fig_att.add_trace(go.Scatter(x=df_cmpp['Date'], y=df_cmpp['Close'], mode='lines',
                                 name='Close', line=dict(color='#d62728', width=3)))
    fig_att.add_trace(go.Scatter(x=df_cmpp['Date'], y=df_cmpp['Adj Close'], mode='lines',
                                 name='Adj Close', line=dict(color='#9467bd', width=3)))
    fig_att.add_trace(go.Scatter(x=df_cmpp['Date'], y=df_cmpp['Volume'], mode='lines',
                                 name='Volume', line=dict(color='#8c564b', width=3)))
    fig_att.update_layout(
        showlegend=True,
        legend={
            'font': {'size': 15}
        },
        xaxis={
            'rangeslider': {'visible': False},
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'showgrid': True,
            'tickfont': {'size': 15},
            'title': {'text': 'Date', 'font': {'size': 20}}
        },
        yaxis={
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'tickfont': {'size': 15},
            'title': {'text': 'Stock Price (in Rupiah)', 'font': {'size': 20}}
        },
        margin={'t': 25, 'b': 0},
        height=500,
        width=500
    )
    st.plotly_chart(
        fig_att, use_container_width=st.session_state.use_container_width)

    # Select the close price column as the target variable
    target_col = 'Close'
    # Create a new dataframe with only the target variable
    target_df = pd.DataFrame(df_cmpp[target_col])
    # Split the data into training and testing sets
    train_size = int(len(target_df) * 0.9)
    train_df = target_df[:train_size]
    test_df = target_df[train_size:]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Define the sequence length and number of features
    sequence_length = 60
    num_features = 1
    # Define the function to create sequences

    def create_sequences(X, y, seq_length):
        X_sequences, y_sequences = [], []
        for i in range(seq_length, len(X)):
            X_sequences.append(X[i-seq_length:i, :])
            y_sequences.append(y[i, :])
        return np.array(X_sequences), np.array(y_sequences)
    # Create training sequences and labels
    X_train, y_train = create_sequences(
        train_scaled, train_scaled, sequence_length)
    # Create testing sequences and labels
    X_test, y_test = create_sequences(
        test_scaled, test_scaled, sequence_length)
    
    if parameter == 'Default':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_lstm_default.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_linear_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_linear_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_linear_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_linear_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_linear_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_linear_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_linear_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_sigmoid_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_tanh_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_tanh_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_tanh_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_tanh_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_tanh_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_tanh_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    else:
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_tanh_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_cmpp.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    st.sidebar.write('Parameter Recommendation: ReLU - Adam')
################################################# SAFE.JK ###############################################
elif ticker == 'SAFE':
    st.image(Image.open('assets/safe-icon.png'), use_column_width=False, width=300)
    st.write('#### Dataset')

    # Read the data
    df_safe = pd.read_excel(
        'transportation_stocks.xlsx', sheet_name='SAFE.JK')

    # Convert the date column to datetime format
    df_safe['Date'] = [datetime.datetime.strptime(str(target_date).split(
        ' ')[0], '%Y-%m-%d').date() for target_date in df_safe['Date']]

    # Fill missing value using mean imputation
    for i in df_safe[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]:
        df_safe[i].fillna(df_safe[i].mean(), inplace=True)

    # Display the dataframe with font size of 14
    st.checkbox('Use container width', value=True, key='use_container_width')
    st.dataframe(
        df_safe, use_container_width=st.session_state.use_container_width)

    st.write(' #### Stock Prices History')
    fig_att = go.Figure()
    fig_att.add_trace(go.Scatter(x=df_safe['Date'], y=df_safe['Open'], mode='lines',
                                 name='Open', line=dict(color='#1f77b4', width=3)))
    fig_att.add_trace(go.Scatter(x=df_safe['Date'], y=df_safe['High'], mode='lines',
                                 name='High', line=dict(color='#ff7f0e', width=3)))
    fig_att.add_trace(go.Scatter(x=df_safe['Date'], y=df_safe['Low'], mode='lines',
                                 name='Low', line=dict(color='#2ca02c', width=3)))
    fig_att.add_trace(go.Scatter(x=df_safe['Date'], y=df_safe['Close'], mode='lines',
                                 name='Close', line=dict(color='#d62728', width=3)))
    fig_att.add_trace(go.Scatter(x=df_safe['Date'], y=df_safe['Adj Close'], mode='lines',
                                 name='Adj Close', line=dict(color='#9467bd', width=3)))
    fig_att.add_trace(go.Scatter(x=df_safe['Date'], y=df_safe['Volume'], mode='lines',
                                 name='Volume', line=dict(color='#8c564b', width=3)))
    fig_att.update_layout(
        showlegend=True,
        legend={
            'font': {'size': 15}
        },
        xaxis={
            'rangeslider': {'visible': False},
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'showgrid': True,
            'tickfont': {'size': 15},
            'title': {'text': 'Date', 'font': {'size': 20}}
        },
        yaxis={
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'tickfont': {'size': 15},
            'title': {'text': 'Stock Price (in Rupiah)', 'font': {'size': 20}}
        },
        margin={'t': 25, 'b': 0},
        height=500,
        width=500
    )
    st.plotly_chart(
        fig_att, use_container_width=st.session_state.use_container_width)

    # Select the close price column as the target variable
    target_col = 'Close'
    # Create a new dataframe with only the target variable
    target_df = pd.DataFrame(df_safe[target_col])
    # Split the data into training and testing sets
    train_size = int(len(target_df) * 0.9)
    train_df = target_df[:train_size]
    test_df = target_df[train_size:]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Define the sequence length and number of features
    sequence_length = 60
    num_features = 1
    # Define the function to create sequences

    def create_sequences(X, y, seq_length):
        X_sequences, y_sequences = [], []
        for i in range(seq_length, len(X)):
            X_sequences.append(X[i-seq_length:i, :])
            y_sequences.append(y[i, :])
        return np.array(X_sequences), np.array(y_sequences)
    # Create training sequences and labels
    X_train, y_train = create_sequences(
        train_scaled, train_scaled, sequence_length)
    # Create testing sequences and labels
    X_test, y_test = create_sequences(
        test_scaled, test_scaled, sequence_length)
    
    if parameter == 'Default':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_lstm_default.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_linear_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_linear_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_linear_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_linear_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_linear_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_linear_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_linear_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/cmpp_relu_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_relu_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_relu_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_relu_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_relu_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_relu_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_relu_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_sigmoid_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_sigmoid_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_sigmoid_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_sigmoid_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_sigmoid_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_sigmoid_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_sigmoid_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_tanh_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_tanh_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_tanh_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_tanh_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_tanh_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_tanh_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    else:
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/safe_tanh_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_safe.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    st.sidebar.write('Parameter Recommendation: ReLU - Nadam')
################################################# SMDR.JK ###############################################
elif ticker == 'SMDR':
    st.image(Image.open('assets/smdr-icon.png'), use_column_width=False, width=600)
    st.write('#### Dataset')

    # Read the data
    df_smdr = pd.read_excel(
        'transportation_stocks.xlsx', sheet_name='SMDR.JK')

    # Convert the date column to datetime format
    df_smdr['Date'] = [datetime.datetime.strptime(str(target_date).split(
        ' ')[0], '%Y-%m-%d').date() for target_date in df_smdr['Date']]

    # # Fill missing value using mean imputation
    # for i in df_smdr[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]:
    #     df_smdr[i].fillna(df_smdr[i].mean(), inplace=True)

    # Display the dataframe with font size of 14
    st.checkbox('Use container width', value=True, key='use_container_width')
    st.dataframe(
        df_smdr, use_container_width=st.session_state.use_container_width)

    st.write(' #### Stock Prices History')
    fig_att = go.Figure()
    fig_att.add_trace(go.Scatter(x=df_smdr['Date'], y=df_smdr['Open'], mode='lines',
                                 name='Open', line=dict(color='#1f77b4', width=3)))
    fig_att.add_trace(go.Scatter(x=df_smdr['Date'], y=df_smdr['High'], mode='lines',
                                 name='High', line=dict(color='#ff7f0e', width=3)))
    fig_att.add_trace(go.Scatter(x=df_smdr['Date'], y=df_smdr['Low'], mode='lines',
                                 name='Low', line=dict(color='#2ca02c', width=3)))
    fig_att.add_trace(go.Scatter(x=df_smdr['Date'], y=df_smdr['Close'], mode='lines',
                                 name='Close', line=dict(color='#d62728', width=3)))
    fig_att.add_trace(go.Scatter(x=df_smdr['Date'], y=df_smdr['Adj Close'], mode='lines',
                                 name='Adj Close', line=dict(color='#9467bd', width=3)))
    fig_att.add_trace(go.Scatter(x=df_smdr['Date'], y=df_smdr['Volume'], mode='lines',
                                 name='Volume', line=dict(color='#8c564b', width=3)))
    fig_att.update_layout(
        showlegend=True,
        legend={
            'font': {'size': 15}
        },
        xaxis={
            'rangeslider': {'visible': False},
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'showgrid': True,
            'tickfont': {'size': 15},
            'title': {'text': 'Date', 'font': {'size': 20}}
        },
        yaxis={
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'tickfont': {'size': 15},
            'title': {'text': 'Stock Price (in Rupiah)', 'font': {'size': 20}}
        },
        margin={'t': 25, 'b': 0},
        height=500,
        width=500
    )
    st.plotly_chart(
        fig_att, use_container_width=st.session_state.use_container_width)

    # Select the close price column as the target variable
    target_col = 'Close'
    # Create a new dataframe with only the target variable
    target_df = pd.DataFrame(df_smdr[target_col])
    # Split the data into training and testing sets
    train_size = int(len(target_df) * 0.9)
    train_df = target_df[:train_size]
    test_df = target_df[train_size:]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Define the sequence length and number of features
    sequence_length = 60
    num_features = 1
    # Define the function to create sequences

    def create_sequences(X, y, seq_length):
        X_sequences, y_sequences = [], []
        for i in range(seq_length, len(X)):
            X_sequences.append(X[i-seq_length:i, :])
            y_sequences.append(y[i, :])
        return np.array(X_sequences), np.array(y_sequences)
    # Create training sequences and labels
    X_train, y_train = create_sequences(
        train_scaled, train_scaled, sequence_length)
    # Create testing sequences and labels
    X_test, y_test = create_sequences(
        test_scaled, test_scaled, sequence_length)
    
    if parameter == 'Default':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_lstm_default.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_linear_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_linear_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_linear_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_linear_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_linear_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_linear_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_linear_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_relu_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_relu_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_relu_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_relu_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_relu_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_relu_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_relu_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_sigmoid_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_sigmoid_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_sigmoid_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_sigmoid_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_sigmoid_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_sigmoid_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_sigmoid_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_tanh_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_tanh_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_tanh_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_tanh_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_tanh_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_tanh_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    else:
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/smdr_tanh_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_smdr.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    st.sidebar.write('Parameter Recommendation: Linear - Adam')
################################################# TMAS.JK ###############################################
elif ticker == 'TMAS':
    st.image(Image.open('assets/tmas-icon.png'), use_column_width=False, width=500)
    st.write('#### Dataset')

    # Read the data
    df_tmas = pd.read_excel(
        'transportation_stocks.xlsx', sheet_name='TMAS.JK')

    # Convert the date column to datetime format
    df_tmas['Date'] = [datetime.datetime.strptime(str(target_date).split(
        ' ')[0], '%Y-%m-%d').date() for target_date in df_tmas['Date']]

    # # Fill missing value using mean imputation
    # for i in df_tmas[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]:
    #     df_tmas[i].fillna(df_tmas[i].mean(), inplace=True)

    # Display the dataframe with font size of 14
    st.checkbox('Use container width', value=True, key='use_container_width')
    st.dataframe(
        df_tmas, use_container_width=st.session_state.use_container_width)

    st.write(' #### Stock Prices History')
    fig_att = go.Figure()
    fig_att.add_trace(go.Scatter(x=df_tmas['Date'], y=df_tmas['Open'], mode='lines',
                                 name='Open', line=dict(color='#1f77b4', width=3)))
    fig_att.add_trace(go.Scatter(x=df_tmas['Date'], y=df_tmas['High'], mode='lines',
                                 name='High', line=dict(color='#ff7f0e', width=3)))
    fig_att.add_trace(go.Scatter(x=df_tmas['Date'], y=df_tmas['Low'], mode='lines',
                                 name='Low', line=dict(color='#2ca02c', width=3)))
    fig_att.add_trace(go.Scatter(x=df_tmas['Date'], y=df_tmas['Close'], mode='lines',
                                 name='Close', line=dict(color='#d62728', width=3)))
    fig_att.add_trace(go.Scatter(x=df_tmas['Date'], y=df_tmas['Adj Close'], mode='lines',
                                 name='Adj Close', line=dict(color='#9467bd', width=3)))
    fig_att.add_trace(go.Scatter(x=df_tmas['Date'], y=df_tmas['Volume'], mode='lines',
                                 name='Volume', line=dict(color='#8c564b', width=3)))
    fig_att.update_layout(
        showlegend=True,
        legend={
            'font': {'size': 15}
        },
        xaxis={
            'rangeslider': {'visible': False},
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'showgrid': True,
            'tickfont': {'size': 15},
            'title': {'text': 'Date', 'font': {'size': 20}}
        },
        yaxis={
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'tickfont': {'size': 15},
            'title': {'text': 'Stock Price (in Rupiah)', 'font': {'size': 20}}
        },
        margin={'t': 25, 'b': 0},
        height=500,
        width=500
    )
    st.plotly_chart(
        fig_att, use_container_width=st.session_state.use_container_width)

    # Select the close price column as the target variable
    target_col = 'Close'
    # Create a new dataframe with only the target variable
    target_df = pd.DataFrame(df_tmas[target_col])
    # Split the data into training and testing sets
    train_size = int(len(target_df) * 0.9)
    train_df = target_df[:train_size]
    test_df = target_df[train_size:]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Define the sequence length and number of features
    sequence_length = 60
    num_features = 1
    # Define the function to create sequences

    def create_sequences(X, y, seq_length):
        X_sequences, y_sequences = [], []
        for i in range(seq_length, len(X)):
            X_sequences.append(X[i-seq_length:i, :])
            y_sequences.append(y[i, :])
        return np.array(X_sequences), np.array(y_sequences)
    # Create training sequences and labels
    X_train, y_train = create_sequences(
        train_scaled, train_scaled, sequence_length)
    # Create testing sequences and labels
    X_test, y_test = create_sequences(
        test_scaled, test_scaled, sequence_length)
    
    if parameter == 'Default':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_lstm_default.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_linear_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_linear_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_linear_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_linear_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_linear_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_linear_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_linear_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_relu_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_relu_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_relu_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_relu_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_relu_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_relu_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_relu_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_sigmoid_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_sigmoid_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_sigmoid_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_sigmoid_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_sigmoid_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_sigmoid_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_sigmoid_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_tanh_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_tanh_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_tanh_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_tanh_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_tanh_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_tanh_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    else:
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/tmas_tanh_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_tmas.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    st.sidebar.write('Parameter Recommendation: Linear - Adam')
################################################# WEHA.JK ###############################################
else:
    st.image(Image.open('assets/weha-icon.png'), use_column_width=False, width=400)
    st.write('#### Dataset')

    # Read the data
    df_weha = pd.read_excel(
        'transportation_stocks.xlsx', sheet_name='WEHA.JK')

    # Convert the date column to datetime format
    df_weha['Date'] = [datetime.datetime.strptime(str(target_date).split(
        ' ')[0], '%Y-%m-%d').date() for target_date in df_weha['Date']]

    # Fill missing value using mean imputation
    for i in df_weha[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]:
        df_weha[i].fillna(df_weha[i].mean(), inplace=True)

    # Display the dataframe with font size of 14
    st.checkbox('Use container width', value=True, key='use_container_width')
    st.dataframe(
        df_weha, use_container_width=st.session_state.use_container_width)

    st.write(' #### Stock Prices History')
    fig_att = go.Figure()
    fig_att.add_trace(go.Scatter(x=df_weha['Date'], y=df_weha['Open'], mode='lines',
                                 name='Open', line=dict(color='#1f77b4', width=3)))
    fig_att.add_trace(go.Scatter(x=df_weha['Date'], y=df_weha['High'], mode='lines',
                                 name='High', line=dict(color='#ff7f0e', width=3)))
    fig_att.add_trace(go.Scatter(x=df_weha['Date'], y=df_weha['Low'], mode='lines',
                                 name='Low', line=dict(color='#2ca02c', width=3)))
    fig_att.add_trace(go.Scatter(x=df_weha['Date'], y=df_weha['Close'], mode='lines',
                                 name='Close', line=dict(color='#d62728', width=3)))
    fig_att.add_trace(go.Scatter(x=df_weha['Date'], y=df_weha['Adj Close'], mode='lines',
                                 name='Adj Close', line=dict(color='#9467bd', width=3)))
    fig_att.add_trace(go.Scatter(x=df_weha['Date'], y=df_weha['Volume'], mode='lines',
                                 name='Volume', line=dict(color='#8c564b', width=3)))
    fig_att.update_layout(
        showlegend=True,
        legend={
            'font': {'size': 15}
        },
        xaxis={
            'rangeslider': {'visible': False},
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'showgrid': True,
            'tickfont': {'size': 15},
            'title': {'text': 'Date', 'font': {'size': 20}}
        },
        yaxis={
            'gridcolor': 'lightgray',
            'gridwidth': 1,
            'tickfont': {'size': 15},
            'title': {'text': 'Stock Price (in Rupiah)', 'font': {'size': 20}}
        },
        margin={'t': 25, 'b': 0},
        height=500,
        width=500
    )
    st.plotly_chart(
        fig_att, use_container_width=st.session_state.use_container_width)

    # Select the close price column as the target variable
    target_col = 'Close'
    # Create a new dataframe with only the target variable
    target_df = pd.DataFrame(df_weha[target_col])
    # Split the data into training and testing sets
    train_size = int(len(target_df) * 0.9)
    train_df = target_df[:train_size]
    test_df = target_df[train_size:]

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    # Define the sequence length and number of features
    sequence_length = 60
    num_features = 1
    # Define the function to create sequences

    def create_sequences(X, y, seq_length):
        X_sequences, y_sequences = [], []
        for i in range(seq_length, len(X)):
            X_sequences.append(X[i-seq_length:i, :])
            y_sequences.append(y[i, :])
        return np.array(X_sequences), np.array(y_sequences)
    # Create training sequences and labels
    X_train, y_train = create_sequences(
        train_scaled, train_scaled, sequence_length)
    # Create testing sequences and labels
    X_test, y_test = create_sequences(
        test_scaled, test_scaled, sequence_length)
    
    if parameter == 'Default':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_lstm_default.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_linear_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_linear_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_linear_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_linear_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_linear_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_linear_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Linear - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_linear_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_relu_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_relu_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_relu_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_relu_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_relu_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_relu_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'ReLU - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_relu_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_sigmoid_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_sigmoid_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_sigmoid_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_sigmoid_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_sigmoid_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_sigmoid_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Sigmoid - AdaMax':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_sigmoid_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Adam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_tanh_adam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaGrad':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_tanh_adagrad.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - Nadam':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_tanh_nadam.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - RMSProp':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_tanh_rmsprop.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - AdaDelta':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_tanh_adadelta.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    elif parameter == 'Tanh - SGD':
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_tanh_sgd.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    else:
        st.write('#### Current Stock Price Prediction')
        # Build LSTM model
        model = load_model('models/weha_tanh_adamax.h5')
        # Generate predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Invert scaling for predictions
        train_predict = scaler.inverse_transform(train_predict)
        train_actual = scaler.inverse_transform(y_train)
        test_predict = scaler.inverse_transform(test_predict)
        test_actual = scaler.inverse_transform(y_test)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(y=test_actual.flatten(),
                                      mode='lines', name='Actual', line=dict(color='red', width=3)))
        fig_pred.add_trace(go.Scatter(y=test_predict.flatten(),
                                      mode='lines', name='Predicted', line=dict(color='green', width=3)))

        fig_pred.update_layout(
            showlegend=True,
            legend={
                'font': {'size': 15}
            },
            xaxis={
                'rangeslider': {'visible': False},
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'showgrid': True,
                'tickfont': {'size': 15},
                'title': {'text': 'Time', 'font': {'size': 20}}
            },
            yaxis={
                'gridcolor': 'lightgray',
                'gridwidth': 1,
                'tickfont': {'size': 15},
                'title': {'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
            },
            margin={'t': 25, 'b': 0},
            height=500,
            width=500
        )
        st.plotly_chart(
            fig_pred, use_container_width=st.session_state.use_container_width)

        st.write('#### Future Stock Price Prediction')
        # Load the data and perform the forecast
        num = 200
        X_Predict = X_test[-num:]
        Forecast = model.predict(X_Predict)
        date = np.array(df_weha.index)
        time = pd.date_range(date[len(date) - 1], periods=num, freq='d')
        df_predict = pd.DataFrame(Forecast, index=time)
        df_predict.columns = ['Forecast']

       # Get current date predictions
        current_date = datetime.datetime(2023, 4, 1).date()
        num = (end_date - start_date).days + 1
        predicted_dates = pd.date_range(start_date, periods=num, freq='d')

        # Visualize the forecast
        fig_future = go.Figure(data=go.Scatter(x=predicted_dates, y=df_predict['Forecast'], mode='lines', line=dict(color='green', width=3)))
        fig_future.update_layout(
            yaxis=dict(
                tickformat='.4s',  # Formats values in millions
                title='Close Price (in Rupiah)',
                tickfont={'size': 12},
                gridcolor='lightgray',
                gridwidth=1
            ),
            xaxis=dict(
                rangeslider={'visible': False},
                gridcolor='lightgray',
                gridwidth=1,
                showgrid=True,
                tickfont={'size': 15},
                title={'text': 'Date', 'font': {'size': 20}}
            ),
            showlegend=False,
            margin={'t': 25, 'b': 0},
            height=500,
            width=500,
        )
        fig_future.update_yaxes(
            gridcolor='lightgray',
            gridwidth=1,
            tickfont={'size': 15},
            title={'text': 'Close Price (in Rupiah)', 'font': {'size': 20}}
        )
        st.plotly_chart(fig_future, use_container_width=st.session_state.use_container_width)
        # Check evaluation metrics
        preds = model.predict(X_test)
        mape = round(metrics.mean_absolute_percentage_error(y_test, preds), 5)
        mae = round(metrics.mean_absolute_error(y_test, preds), 5)
        mse = round(metrics.mean_squared_error(y_test, preds), 5)
        rmse = round(math.sqrt(mse), 5)
        r2 = round(metrics.r2_score(y_test, preds), 2)

        st.write('##### Evaluation Metrics :')
        st.write(
            f'<span style="font-size:20px">Mean Absolute Percentage Error (MAPE) : {mape}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Absolute Error (MAE) : {mae}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Mean Squared Error (MSE) : {mse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">Root Mean Squared Error (RMSE) : {rmse}</span>', unsafe_allow_html=True)
        st.write(
            f'<span style="font-size:20px">R-Squared (R2) : {r2}</span>', unsafe_allow_html=True)
    st.sidebar.write('Parameter Recommendation: Linear - Adam')
# Let's remove the footer watermark
hide_streamlit_style = '''
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            '''
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Set the footer style
footer = """
<div style='text-align: center; font-size: 14px; padding: 10px;'>
    Made with ❤️ by <a href="https://github.com/Bayunova28" target="_blank" style="color: black;">Willibrordus Bayu</a>
</div>
"""
# Display the footer
st.markdown(footer, unsafe_allow_html=True)
