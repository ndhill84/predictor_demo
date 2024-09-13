import streamlit as st
import numpy as np
np.float_ = np.float64
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots

st.title('Spyder Price Predictor')


# Data fetching
start_date = "2016-01-01" #Start date for data fetching
today = date.today().strftime("%Y-%m-%d")

stocks = ('NVDA', 'AAPL', 'GOOGL', 'MSFT', 'TSLA')
selected_stock = st.selectbox('Select dataset for prediction', stocks) #selector drop down box

n_years = st.slider('Months of prediction:', 6, 60, 36)
period = n_years * 30 #Period * days (30 = months, 365 = years, 7 = weeks)

def load_data(ticker):
    """
    Downloads stock data from Yahoo Finance and loads it into a Pandas DataFrame.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL' for Apple).

    Returns
    -------
    data : pandas.DataFrame
        Contains the stock data, with columns for 'Date', 'Open', 'High',
        'Low', 'Close', 'Adj Close', and 'Volume'.
    """
    data = yf.download(ticker, start_date
, today)
    data.reset_index(inplace=True)
    print(data)
    return data

data_load_state = st.text('Loading')
data = load_data(selected_stock)
data100 = data[:100]
data_load_state.text('Done..!')

st.subheader(selected_stock)

def plot_raw_data():
    """
    Plot the raw data as a candlestick chart with a secondary axis
    for volume.
    """
    fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Candlestick(x=data100['Date'], open=data100['Open'], high=data100['High'], low=data100['Low'], close=data100['Close']),secondary_y=True, exclude_empty_subplots=False)# xaxis_range=['2024-06-01', data['Date'].max()])
    fig.add_trace(go.Bar(x=data100['Date'], y=data100['Volume']),secondary_y=False)
    fig.layout.yaxis2.showgrid = False
    #fig.layout.update(title_text=str(selected_stock), xaxis_rangeslider_visible=True, xaxis_title="Date", yaxis_title="Price" xaxis_range=)
    st.plotly_chart(fig)
    
    # candlesticks = go.Candlestick(
    #     x=data100['Date'],
    #     open=data100['Open'],
    #     high=data100['High'],
    #     low=data100['Low'],
    #     close=data100['Close'],
    #     showlegend=False
    # )
    # volume_bars = go.Bar(
    #     x=data100['date'],
    #     y=data100['volume'],
    #     showlegend=False,
    #     marker={
    #     "color": "rgba(128,128,128,0.5)",
    #     }
    # )
    # fig2 = go.Figure(candlesticks)
    # fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    # fig2.add_trace(candlesticks, secondary_y=True)
    # fig2.add_trace(volume_bars, secondary_y=False)
    
    # fig.update_layout(
    #     title=str(selected_stock), height=800,
    #     xaxis={"rangeslider": {"visible": False}},
    # )
    # fig2.update_yaxes(title="Price $", secondary_y=True, showgrid=True)
    # fig2.update_yaxes(title="Volume", secondary_y=False, showgrid=True)
    # #st.plotly_chart(fig2)
    


def plot_forecast():
    """
    Plot forecast using Prophet.
    """
    # Prepare data for Prophet model
    df_train = data.rename(columns={"Date": "ds", "Close": "y"})

    # Create and fit Prophet model
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
    m.fit(df_train)

    # Generate future dates
    future = m.make_future_dataframe(periods=period)

    # Generate forecast
    forecast = m.predict(future)

    # Plot forecast
    st.subheader("Forecast")
    st.plotly_chart(plot_plotly(m, forecast))

    # Plot components of forecast
    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)
    
    
plot_forecast()

plot_raw_data()

st.write(data100.tail(100)) #Prints the data table at the bottom of the page. Limited results to 100