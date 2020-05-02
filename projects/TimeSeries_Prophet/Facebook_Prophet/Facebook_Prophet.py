#!/usr/bin/env python
# coding: utf-8

#https://github.com/gumdropsteve/intro_to_prophet
# In[1]:


import warnings
import numpy as np
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric


# In[2]:


# load walmart ($wmt) historical stock data
wmt = pd.read_csv('C:\\Users\\ankur\\Downloads\\WMT.csv')
wmt.head()


# In[3]:


wmt.info()


# In[4]:


wmt.describe()


# In[5]:


#  new dataframe with date and adjusted closing price
wmt = wmt[['Date','Adj Close']]
wmt.head()


# In[6]:


# adjust column names for prophet compliance
wmt.columns = ['ds','y']
wmt.head()


# In[7]:


# convert dates from string to datetime
wmt.ds = pd.to_datetime(wmt.ds)

# display adjustments
wmt.tail()


# In[8]:


# frame up w/ grid
plt.figure(figsize=(16,4))
plt.grid(linestyle='-.')

# sketch in data
plt.plot(wmt.ds, wmt.y, 'b')

# set title & labels
plt.title('$WMT Adj. Closing', fontsize=18)
plt.ylabel('Price ($)', fontsize=13)
plt.xlabel('Time (year)', fontsize=13)

# display graph
plt.show()


# In[9]:


# set prophet model 
prophet = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False,)
# fit $wmt data to model
prophet.fit(wmt)
# build future dataframe for 5 years
build_forecast = prophet.make_future_dataframe(periods=365*5, freq='D')
print(build_forecast.tail())
# forecast future df with model
forecast = prophet.predict(build_forecast)

# plot forecasts
prophet.plot(forecast, xlabel='Date', ylabel='Share Price ($)')
plt.title('Walmart Stock Price ($WMT)')
plt.show()


# In[10]:


# tell us more about the forecast
components = prophet.plot_components(forecast)


# In[11]:


# narrow selection to dates past initial dataframe
future_preds = forecast.loc[forecast.ds>'2019-07-02']
# select date, prediction, lower and upper limits 
future_preds = future_preds[['ds','yhat','yhat_lower','yhat_upper']]
# display some predictions
future_preds.sample(5)


# In[ ]:


# cross validate 1 year every half year at 30 years
wmt_cv = cross_validation(prophet, initial='10950 days', period='180 days', horizon = '365 days')
wmt_cv.head()


# In[ ]:


#measure performance
wmt_pm = performance_metrics(wmt_cv)


# In[ ]:


# final accuracy
wmt_pm.tail(3)


# In[ ]:


# visualize mape across horizon (continued sort)
fig = plot_cross_validation_metric(wmt_cv, metric='mape')


# In[ ]:


# where did prophet identify changepoints
prophet.changepoints

