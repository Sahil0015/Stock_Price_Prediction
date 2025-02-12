import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf



st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")


end_time = datetime.now()
start_time = datetime(end_time.year - 20, end_time.month, end_time.day)


google_data = yf.download(stock, start_time, end_time)
google_data.columns = [' '.join(col).strip() for col in google_data.columns.values]
google_data.columns = google_data.columns.str.replace(r'\s.*$', '', regex=True)
print(google_data.columns)


short_term_trading_model = load_model(
    "short_term_model.h5",
    custom_objects={"mse": "mean_squared_error"}
)

medium_term_trading_model = load_model(
    "medium_term_model.h5",
    custom_objects={"mse": "mean_squared_error"}
)

long_term_trading_model = load_model(
    "long_term_model.h5",
    custom_objects={"mse": "mean_squared_error"}
)


st.subheader("Stock Data")
st.write(google_data)


short_splitting_len = int(len(google_data) * 0.8)
short_X_text = pd.DataFrame(google_data.Close[short_splitting_len:])

medium_splitting_len = int(len(google_data) * 0.75)
medium_X_text = pd.DataFrame(google_data.Close[medium_splitting_len:])

long_splitting_len = int(len(google_data) * 0.7)
long_X_text = pd.DataFrame(google_data.Close[long_splitting_len:])

def plot_graph(figsize, values, full_data):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    return fig

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_Days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_Days'], google_data))

st.subheader('Original Close Price and MA for 40 days')
google_data['MA_for_40_Days'] = google_data.Close.rolling(40).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_40_Days'], google_data))

st.subheader('Original Close Price and MA for 10 days')
google_data['MA_for_10_Days'] = google_data.Close.rolling(10).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_10_Days'], google_data))


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_short_data = scaler.fit_transform(short_X_text[['Close']])
scaled_medium_data = scaler.fit_transform(medium_X_text[['Close']])
scaled_long_data = scaler.fit_transform(long_X_text[['Close']])


short_term_x_data = []
short_term_y_data = []

medium_term_x_data = []
medium_term_y_data = []

long_term_x_data = []
long_term_y_data = []

for i in range(10, len(scaled_short_data)):
  short_term_x_data.append(scaled_short_data[i-10:i])
  short_term_y_data.append(scaled_short_data[i])

for i in range(40, len(scaled_medium_data)):
  medium_term_x_data.append(scaled_medium_data[i-40:i])
  medium_term_y_data.append(scaled_medium_data[i])

for i in range(100, len(scaled_long_data)):
  long_term_x_data.append(scaled_long_data[i-100:i])
  long_term_y_data.append(scaled_long_data[i])


short_term_x_data = np.array(short_term_x_data)
short_term_y_data = np.array(short_term_y_data)

medium_term_x_data = np.array(medium_term_x_data)
medium_term_y_data = np.array(medium_term_y_data)

long_term_x_data = np.array(long_term_x_data)
long_term_y_data = np.array(long_term_y_data)


short_term_predictions = short_term_trading_model.predict(short_term_x_data)
inv_short_term_predictions = scaler.inverse_transform(short_term_predictions)
inv_short_y_test = scaler.inverse_transform(short_term_y_data)

medium_term_predictions = medium_term_trading_model.predict(medium_term_x_data)
inv_medium_term_predictions = scaler.inverse_transform(medium_term_predictions)
inv_medium_y_test = scaler.inverse_transform(medium_term_y_data)

long_term_predictions = long_term_trading_model.predict(long_term_x_data)
inv_long_term_predictions = scaler.inverse_transform(long_term_predictions)
inv_long_y_test = scaler.inverse_transform(long_term_y_data)

st.subheader("Original values vs Predicted values")
plotting_short_term_data = pd.DataFrame(
    {
        'Actual': inv_short_y_test.reshape(-1),
        'Predicted': inv_short_term_predictions.reshape(-1)
    } ,
    index=google_data.index[short_splitting_len + 10:]
)
st.write("Short Trading")
st.write(plotting_short_term_data)

# Short Trading plot
fig1 = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:short_splitting_len+10], plotting_short_term_data], axis=0))
plt.legend(["Train Data", "Original Test data", "Predicted Test data"])
st.pyplot(fig1)


plotting_medium_term_data = pd.DataFrame(
    {
        'Actual': inv_medium_y_test.reshape(-1),
        'Predicted': inv_medium_term_predictions.reshape(-1)
    } ,
    index=google_data.index[medium_splitting_len + 40:]
)
st.write("Medium Trading")
st.write(plotting_medium_term_data)

# Medium Trading plot
fig2 = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:medium_splitting_len+40], plotting_medium_term_data], axis=0))
plt.legend(["Train Data", "Original Test data", "Predicted Test data"])
st.pyplot(fig2)


plotting_long_term_data = pd.DataFrame(
    {
        'Actual': inv_long_y_test.reshape(-1),
        'Predicted': inv_long_term_predictions.reshape(-1)
    } ,
    index=google_data.index[long_splitting_len + 100:]
)
st.write("Long Trading")
st.write(plotting_long_term_data)

# Long Trading plot
fig3 = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:long_splitting_len+100], plotting_long_term_data], axis=0))
plt.legend(["Train Data", "Original Test data", "Predicted Test data"])
st.pyplot(fig3)