import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.markdown("<h1 style='text-align: center; font-size: 2em;'>Projecting the Philippine Labor Force</h1>", unsafe_allow_html=True)
about, general, hier = st.tabs(['About the Project', 'General Forecast', 'Forecast by Industry'])

df = pd.read_csv('labor_force.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df.index.freq = 'QS-OCT'
for column in df:
  df[column] = df[column].astype('int')

df_byindustry = pd.read_csv('labor_force_by_major_industry.csv')
df_byindustry['Date'] = pd.to_datetime(df_byindustry['Date'])
df_byindustry = df_byindustry.set_index('Date')
df_byindustry.index.freq = 'QS-OCT'
for column in df_byindustry:
  df_byindustry[column] = df_byindustry[column].astype('int')
  
with about:
  st.markdown("""
        <div style="text-align: justify;">
            <h3>Project Description</h3>
            <p>
                This project aims to use a simple model to forecast the Philippine Labor Force Population during the forecast period. The overall forecast uses the Autoregressive integrated Moving Averages (ARIMA) model while the forecast by industry uses either the ARIMA model or naive forecast depending on the better performing model.
            </p>
            <p>
                Feel free to explore the other tabs to see the forecast.
            </p>
        </div>
    """, unsafe_allow_html=True)

with general:
  st.subheader('Philippine Labor Force Population Forecast')
  with st.expander('Forecast Period'):
    start_date = st.date_input('Forecast Period Start Date:', 
                           min_value=datetime(2017, 10, 1), 
                           max_value=datetime(2020, 10, 1), 
                           value=datetime(2017, 10, 1)
                           )
    
    end_date = st.date_input('Forecast Period End Date:', 
                           min_value=datetime(2021, 10, 1), 
                           max_value=datetime(2025, 10, 1), 
                           value=datetime(2021, 10, 1)
                           )

  train_df = df['2003-01-01':start_date]
  train_df.index.freq = 'QS-OCT'
  test_df = df[start_date:'2021-10-01']
  test_df.index.freq = 'QS-OCT'
  new_daterange = pd.date_range(start=start_date, end=end_date, freq='QS-OCT')
  new_df = pd.DataFrame(index=new_daterange)
  test_df = pd.concat([test_df, new_df])
  test_df['Labor Force Total'].fillna(method='ffill', inplace=True)
  duplicate_index = test_df.index.duplicated()
  test_df = test_df[~duplicate_index]

  model = ARIMA(train_df, order=(8, 1, 8), freq='QS-OCT')
  model_fit = model.fit()

  naive_forecast = test_df.shift(1)[start_date:end_date]
  arima_forecast = pd.DataFrame(model_fit.forecast(len(test_df), dynamic=True).values, index=test_df.index,\
                         columns=['Labor Force Total'])

  fig = px.line()
  fig.add_scatter(x=train_df.index, y=train_df['Labor Force Total'], mode='lines', name='train', line=dict(color='blue'))
  fig.add_scatter(x=test_df.index, y=test_df['Labor Force Total'], mode='lines', name='test', line=dict(color='gray'))
  fig.add_scatter(x=naive_forecast.index, y=naive_forecast['Labor Force Total'], mode='lines', name='naive', line=dict(color='red'))
  fig.add_scatter(x=arima_forecast.index, y=arima_forecast['Labor Force Total'], mode='lines', name='ARIMA', line=dict(color='orange'))

  fig.update_layout(xaxis_title='Date',
                    yaxis_title='Philippine Labor Force Population (in thousands)',
                    showlegend=True)

  st.plotly_chart(fig)

  arima_rmse = np.sqrt(mean_squared_error(test_df['Labor Force Total'], arima_forecast['Labor Force Total']))
  arima_mape = np.mean(np.abs((test_df['Labor Force Total'] - arima_forecast['Labor Force Total']) / test_df['Labor Force Total'])) * 100

  delta_pct = arima_forecast.pct_change() * 100
  mean_delta_pct = np.mean(delta_pct)

  color = 'rgb(0, 255, 0)' if mean_delta_pct >= 0 else 'rgb(255, 0, 0)'
  mean_delta_pct_color = f'<span style="color:{color};">{mean_delta_pct.round(2)}%</span>'

  st.markdown(f'Average Delta: {mean_delta_pct_color}', unsafe_allow_html=True)
  with st.expander('Error Metrics'):
    st.write(f'Root Mean Squared Error: {arima_rmse.round(2)}')
    st.write(f'Mean Average Percentage Error: {arima_mape.round(2)}%')

with hier:
  train_df_byindustry = df_byindustry['2003-01-01':'2017-10-01']
  train_df_byindustry.index.freq = 'QS-OCT'
  test_df_byindustry = df_byindustry['2017-10-01':'2021-10-01']
  test_df_byindustry.index.freq = 'QS-OCT'

  naive_forecast_byindustry = pd.DataFrame(index=test_df_byindustry.index)
  arima_forecast_byindustry = pd.DataFrame(index=test_df_byindustry.index)
  industry = df_byindustry.columns.tolist()
  arima_industry = [
    (4,1,10),
    (4,1,6),
    (4,1,10),
    (1,1,8),
    (2,1,6),
    (1,1,10),
    (4,1,10),
    (16,1,8),
    (8,1,8),
    (4,1,12),
    (12,1,6),
    (16,1,8),
    (5,1,5),
    (20,1,20),
    (15,1,1),
    (15,1,5),
    (7,1,5),
    (8,1,8),
    (6,1,6),
    (7,1,11),
  ]
  
  n = 0
  
  st.subheader('Forecast Period: January 2018 to October 2021')
  
  if 'expand_all' not in st.session_state:
    st.session_state.expand_all = False
  if st.button('Expand/Collapse All'):
    st.session_state.expand_all = not st.session_state.expand_all
    
  for column in df_byindustry:
    with st.expander(column, expanded=getattr(state, "expand_all", False)):
      model_byindustry = ARIMA(train_df_byindustry[industry[n]], order=arima_industry[n], freq='QS-OCT')
      model_fit_byindustry = model_byindustry.fit()
      
      naive_forecast_byindustry[industry[n]] = df_byindustry[industry[n]].shift(1)['2017-10-01':'2021-10-01']
      arima_forecast_byindustry[industry[n]] = model_fit_byindustry.forecast(len(test_df_byindustry),dynamic=True).values
      
      fig = px.line()
      fig.add_scatter(x=train_df_byindustry.index, y=train_df_byindustry[industry[n]], mode='lines', name='train', line=dict(color='blue'))
      fig.add_scatter(x=test_df_byindustry.index, y=test_df_byindustry[industry[n]], mode='lines', name='test', line=dict(color='gray'))
      fig.add_scatter(x=naive_forecast_byindustry.index, y=naive_forecast_byindustry[industry[n]], mode='lines', name='naive', line=dict(color='red'))
      fig.add_scatter(x=arima_forecast_byindustry.index, y=arima_forecast_byindustry[industry[n]], mode='lines', name='ARIMA', line=dict(color='orange'))
      
      fig.update_layout(xaxis_title='Date',
                        yaxis_title='Philippine Labor Force Population (in thousands)',
                        showlegend=True)
      st.plotly_chart(fig)
      
      arima_rmse = np.sqrt(mean_squared_error(test_df_byindustry[industry[n]], arima_forecast_byindustry[industry[n]]))
      arima_mape = np.mean(np.abs((test_df_byindustry[industry[n]] - arima_forecast_byindustry[industry[n]]) / test_df_byindustry[industry[n]])) * 100
      naive_rmse = np.sqrt(mean_squared_error(test_df_byindustry[industry[n]], naive_forecast_byindustry[industry[n]]))
      naive_mape = np.mean(np.abs((test_df_byindustry[industry[n]] - naive_forecast_byindustry[industry[n]]) / test_df_byindustry[industry[n]])) * 100
      
      chosen_model = 'ARIMA Forecast' if arima_mape < naive_mape else 'Naive Forecast'
      
      delta_pct_byindustry = arima_forecast_byindustry[industry[n]].pct_change() * 100 if chosen_model == 'ARIMA Forecast' else naive_forecast_byindustry[industry[n]].pct_change() * 100
      mean_delta_pct_byindustry = np.mean(delta_pct_byindustry)
      color = 'rgb(0, 255, 0)' if mean_delta_pct_byindustry >= 0 else 'rgb(255, 0, 0)'
      mean_delta_pct_byindustry_color = f'<span style="color:{color};">{mean_delta_pct_byindustry.round(2)}%</span>'

      actual_delta_pct = test_df_byindustry[industry[n]].pct_change() * 100
      actual_mean_delta_pct = np.mean(actual_delta_pct)
      actual_color = 'rgb(0, 255, 0)' if actual_mean_delta_pct >= 0 else 'rgb(255, 0, 0)'
      actual_mean_delta_pct_color = f'<span style="color:{actual_color};">{actual_mean_delta_pct.round(2)}%</span>'
      
      st.write(f'Best Model: {chosen_model}')
      st.markdown(f'Model Average Delta: {mean_delta_pct_byindustry_color}', unsafe_allow_html=True)
      st.markdown(f'Actual Average Delta: {actual_mean_delta_pct_color}', unsafe_allow_html=True)
      
      n +=1
