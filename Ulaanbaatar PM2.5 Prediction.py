
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Ulaanbaatar-PM2.5-Prediction" data-toc-modified-id="Ulaanbaatar-PM2.5-Prediction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ulaanbaatar PM2.5 Prediction</a></span></li><li><span><a href="#Importing-data-and-assumptions" data-toc-modified-id="Importing-data-and-assumptions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Importing data and assumptions</a></span></li><li><span><a href="#Visualizing-Features" data-toc-modified-id="Visualizing-Features-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Visualizing Features<a name="visualize"></a></a></span><ul class="toc-item"><li><span><a href="#Determinations-made-from-visualizations" data-toc-modified-id="Determinations-made-from-visualizations-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Determinations made from visualizations</a></span></li></ul></li><li><span><a href="#Clean-Data" data-toc-modified-id="Clean-Data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Clean Data</a></span><ul class="toc-item"><li><span><a href="#Drop-unnecessary-features" data-toc-modified-id="Drop-unnecessary-features-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Drop unnecessary features</a></span></li><li><span><a href="#Value-vs-AQI" data-toc-modified-id="Value-vs-AQI-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Value vs AQI</a></span></li><li><span><a href="#Drop-extra-date/time-columns" data-toc-modified-id="Drop-extra-date/time-columns-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Drop extra date/time columns</a></span></li><li><span><a href="#Create-canonical-date-feature" data-toc-modified-id="Create-canonical-date-feature-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Create canonical date feature</a></span></li><li><span><a href="#Handling-outliers" data-toc-modified-id="Handling-outliers-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Handling outliers</a></span></li><li><span><a href="#Handle-NaNs" data-toc-modified-id="Handle-NaNs-4.6"><span class="toc-item-num">4.6&nbsp;&nbsp;</span>Handle NaNs</a></span></li><li><span><a href="#Handle-990-DIR-Values" data-toc-modified-id="Handle-990-DIR-Values-4.7"><span class="toc-item-num">4.7&nbsp;&nbsp;</span>Handle 990 DIR Values</a></span></li></ul></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Convert-cyclical-&amp;-circular-features" data-toc-modified-id="Convert-cyclical-&amp;-circular-features-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Convert cyclical &amp; circular features</a></span><ul class="toc-item"><li><span><a href="#DIR" data-toc-modified-id="DIR-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>DIR</a></span></li><li><span><a href="#Hour-and-Month---Convert-cyclical-feature" data-toc-modified-id="Hour-and-Month---Convert-cyclical-feature-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Hour and Month - Convert cyclical feature</a></span></li></ul></li><li><span><a href="#Convert-Value-Field-from-mg-to-µg" data-toc-modified-id="Convert-Value-Field-from-mg-to-µg-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Convert Value Field from mg to µg</a></span></li><li><span><a href="#Convert-TEMP-and-DEWP-from-F-to-C" data-toc-modified-id="Convert-TEMP-and-DEWP-from-F-to-C-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Convert TEMP and DEWP from F to C</a></span></li><li><span><a href="#Future-Actions-for-Feature-Engineering" data-toc-modified-id="Future-Actions-for-Feature-Engineering-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Future Actions for Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Previous-hour-value-feature." data-toc-modified-id="Previous-hour-value-feature.-5.4.1"><span class="toc-item-num">5.4.1&nbsp;&nbsp;</span>Previous hour value feature.</a></span></li><li><span><a href="#Moving-average-feature" data-toc-modified-id="Moving-average-feature-5.4.2"><span class="toc-item-num">5.4.2&nbsp;&nbsp;</span>Moving average feature</a></span></li><li><span><a href="#Number-of-days-since-start" data-toc-modified-id="Number-of-days-since-start-5.4.3"><span class="toc-item-num">5.4.3&nbsp;&nbsp;</span>Number of days since start</a></span></li></ul></li><li><span><a href="#Get-final-dataset" data-toc-modified-id="Get-final-dataset-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Get final dataset</a></span></li></ul></li><li><span><a href="#Split-into-Training-and-Test-Data" data-toc-modified-id="Split-into-Training-and-Test-Data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Split into Training and Test Data</a></span><ul class="toc-item"><li><span><a href="#Prepare-Data-Set-for-Regression" data-toc-modified-id="Prepare-Data-Set-for-Regression-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Prepare Data Set for Regression</a></span></li><li><span><a href="#Prepare-Data-Set-for-Classification" data-toc-modified-id="Prepare-Data-Set-for-Classification-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Prepare Data Set for Classification</a></span></li></ul></li><li><span><a href="#Implement-ML-Algorithms" data-toc-modified-id="Implement-ML-Algorithms-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Implement ML Algorithms</a></span><ul class="toc-item"><li><span><a href="#Regression" data-toc-modified-id="Regression-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Regression</a></span><ul class="toc-item"><li><span><a href="#Linear-Regression" data-toc-modified-id="Linear-Regression-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>Linear Regression</a></span></li><li><span><a href="#Neural-Network-Regression" data-toc-modified-id="Neural-Network-Regression-7.1.2"><span class="toc-item-num">7.1.2&nbsp;&nbsp;</span>Neural Network Regression</a></span></li><li><span><a href="#Decision-Forest" data-toc-modified-id="Decision-Forest-7.1.3"><span class="toc-item-num">7.1.3&nbsp;&nbsp;</span>Decision Forest</a></span></li><li><span><a href="#Decision-Tree-+-AdaBoost" data-toc-modified-id="Decision-Tree-+-AdaBoost-7.1.4"><span class="toc-item-num">7.1.4&nbsp;&nbsp;</span>Decision Tree + AdaBoost</a></span></li></ul></li><li><span><a href="#Classification" data-toc-modified-id="Classification-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Classification</a></span><ul class="toc-item"><li><span><a href="#Multiclass-neural-network" data-toc-modified-id="Multiclass-neural-network-7.2.1"><span class="toc-item-num">7.2.1&nbsp;&nbsp;</span>Multiclass neural network</a></span></li><li><span><a href="#Multiclass-logistic-regression" data-toc-modified-id="Multiclass-logistic-regression-7.2.2"><span class="toc-item-num">7.2.2&nbsp;&nbsp;</span>Multiclass logistic regression</a></span></li><li><span><a href="#Multiclass-decision-forest" data-toc-modified-id="Multiclass-decision-forest-7.2.3"><span class="toc-item-num">7.2.3&nbsp;&nbsp;</span>Multiclass decision forest</a></span></li><li><span><a href="#Multiclass-decision-jungle" data-toc-modified-id="Multiclass-decision-jungle-7.2.4"><span class="toc-item-num">7.2.4&nbsp;&nbsp;</span>Multiclass decision jungle</a></span></li></ul></li></ul></li><li><span><a href="#Evaluate-Models" data-toc-modified-id="Evaluate-Models-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Evaluate Models</a></span></li><li><span><a href="#To-Do" data-toc-modified-id="To-Do-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>To Do</a></span></li></ul></div>

# # Ulaanbaatar PM2.5 Prediction
# 
# The purpose of this notebook is to create a predictive model of PM2.5 levels in Ulaanbaatar, Mongolia. 

# # Importing data and assumptions
# 
# The data exists in a CSV file that has the AQI data from the US Embassy in Ulaanbaatar and weather data from the Buyant Uhaa weather station. 
# 
# A few key facts about the AQI data:
# - AQI contains only PM2.5 data
# - AQI above 500 may not always be captured as the US AQI scale officially only goes to 500.
# 
# A few key facts about the weather data:
# - The weather station is located approximately 30 kilometers away from the air quality station
# - The location of the weather station is in the westernmost part of the valley Ulaanbaatar sits in

# In[3]:


# Import relevant items
import pandas as pd
import numpy as np


# In[4]:


# Let's first load the data and take a look at what we have.
df = pd.read_csv('weather-and-aqi-v4.csv')


# The head of the dataframe shows lots of columns and LOTS of NaN's.

# In[5]:


print(df.head())
print(df.columns)


# In[6]:


df.dtypes


# There are a large number of columns that are unneccesary. There are duplicate columns for Date, Year, Month, Day, and Hour. There are also columns for location name, station id, units, and intervals of measurement that are not useful in analysis. For now we can leave these columns in. 

# In[7]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [16,9]


# # Visualizing Features<a name="visualize"></a>
# 
# **Let's plot the various features (pollution level, time, month, wind speed, etc) to find any relationships.**
# 
# When plotting the PM2.5 concentration by month you can clearly see that winter months have a much larger variation in pollution levels, including some very high levels.

# In[8]:


x = df['Month']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('Month')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Month')
plt.show()


# Looking at the value plot by hour you can see there are two spikes each day, one between 9-11AM and the other starting around 20 and continuing through the night until 4.

# In[9]:


x = df['Hour']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('Hour')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Hour')
plt.show()


# At higher windspeeds you notice a big drop in the recorded PM2.5 levels. It seems wind speed may be a good feature to predict PM2.5.

# In[10]:


x = df['SPD']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('SPD')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Windspeed')
plt.show()


# Make facet or subplot of PM2.5 levels by windspeed with one subplot per month. This will show if these lower measured values at low windspeeds are evenly distrbuted throughout the year or if they are mostly in certain months.

# In[11]:


# Create a dataframe for each month. Less thinking than slicing it for each visualization. Could do this with a function also
month1 = df[df.Month == 1]
month2 = df[df.Month == 2]
month3 = df[df.Month == 3]
month4 = df[df.Month == 4]
month5 = df[df.Month == 5]
month6 = df[df.Month == 6]
month7 = df[df.Month == 7]
month8 = df[df.Month == 8]
month9 = df[df.Month == 9]
month10 = df[df.Month == 10]
month11 = df[df.Month == 11]
month12 = df[df.Month == 12]


# In[12]:


x = month1['SPD']
y = month1['AQI']
plt.scatter(x,y, color='blue')
plt.xlabel('Wind Speed')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Windspeed - January')
plt.show()


# In[13]:


x = month1['TEMP']
y = month1['Value']
plt.scatter(x,y)
plt.xlabel('Temperature')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Temperature - January')
plt.show()


# In[14]:


x = month1['TEMP']
y = month1['SPD']
col = np.where(month1['AQI']<100,'None',np.where(month1['AQI']>100,'red','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Temperature')
plt.ylabel('Wind Speed')
plt.title('PM2.5 > 100 by Temperature and Windspeed - January')


# In[15]:


plt.show()


# In[16]:


x = month1['TEMP']
y = month1['SPD']
col = np.where(month1['AQI']<100,'green',np.where(month1['AQI']>100,'None','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Temperature')
plt.ylabel('Wind Speed')
plt.title('PM2.5 < 100 by Temperature and Windspeed - January')
plt.show()


# In[17]:


x = month1['Hour']
y = month1['TEMP']
col = np.where(month1['AQI']<100,'green',np.where(month1['AQI']>100,'None','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Hour')
plt.ylabel('Temperature')
plt.title('PM2.5 < 100 by Temperature and Hour - January')
plt.show()


# In[18]:


x = month1['DIR']
y = month1['SPD']
col = np.where(month1['AQI']<100,'None',np.where(month1['AQI']>100,'red','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Wind Direction')
plt.ylabel('Wind Speed')
plt.title('PM2.5 by Windspeed and Direction - January')
plt.xlim(0, 360)
plt.show()


# ## Determinations made from visualizations
# 
# - It is clear that at higher windspeed AQI is lower. Wind speed is possibly a good predictor.
# - Winter months are a large predictor of AQI. Month is possibly a good predictor.
# - While definite spikes are seen at certain times of day (from 5-11AM), values at all times of day have a high variance. Nevertheless it does seem somewhat predictive.
# - Wind direction is given in a 360 degree arc, but variable wind directions are given as 990.
# - Hours are given on a range from 0-24
# 
# == Next Steps ==
# - Remove unneeded features (those that will be unavailable as inputs) - cleaning
# - Find which features have NaNs, determine what to do with them - cleaning
# - Change direction and hour of day to something machine learning models will understand better (more on this later) - feature engineering
# - Determine how to handle dates - feature engineering

# # Clean Data
# ## Drop unnecessary features

# Thus far we have looked at the data available and relationships of some key features. Another key factor is determining what input data will be available to predict PM2.5. This model aims to predict PM2.5 levels into the future. This is possible due to weather forecasting providing inputs for the model. As such we will be limited to the data that is available from the weather forecast provider. Data that is available from several weather API forecasters:
# 
# - Temperature
# - Humidity
# - Wind speed
# - Wind direction
# - Dew point
# 
# To start, let's drop columns that clearly have no value as features.

# In[19]:


df.columns


# Source.Name, Site, Parameter, Unit, Duration, USAF, and WBAN are the same for every row, and as such are not useful features. These are site identifiers for the weather station, the duration of measurement (which is constant), and the unit of PM2.5 measurement (which is in milligrams per cubic meter).

# In[20]:


df = df.drop(['Source.Name', 'Site', 'Parameter', 'Unit', 'Duration', 'USAF', 'WBAN'], axis=1)


# Date Key.1, Year.1, Month.1, Day.1, and Hour.1 are duplicates of the original date features. These were used to create the date key that was then used to combine the PM2.5 and weather data sets. Removing these will cause no harm.

# In[21]:


df = df.drop(['Date Key.1', 'Year.1', 'Month.1', 'Day.1', 'Hour.1'], axis=1)


# In[22]:


df.columns


# The remaining fields are either 1) date fields, 2) PM2.5 values or their derived AQI values, and 3) weather data. As noted before we will only have a few features available for use in prediction. As such we will remove the rest. Below is an inventory of the available weather features. Those marked in bold will be removed
# 
# - **DIR - WIND DIRECTION IN COMPASS DEGREES, 990 = VARIABLE, REPORTED AS** "***" WHEN AIR IS CALM (SPD WILL THEN BE 000)
# - **SPD** & GUS = **WIND SPEED** & GUST IN MILES PER HOUR 
# - CLG = CLOUD CEILING--LOWEST OPAQUE LAYER
# - SKC = SKY COVER
# - L = LOW CLOUD TYPE, SEE BELOW
# - M = MIDDLE CLOUD TYPE, SEE BELOW
# - H = HIGH CLOUD TYPE, SEE BELOW 
# - VSB = VISIBILITY IN STATUTE MILES TO NEAREST TENTH 
# - MW MW1 MW2 MW3 = MANUALLY OBSERVED PRESENT WEATHER--LISTED BELOW IN PRESENT WEATHER TABLE
# - AW AW1 AW2 AW3 = AUTO-OBSERVED PRESENT WEATHER--LISTED BELOW IN PRESENT WEATHER TABLE
# - W = PAST WEATHER INDICATOR, SEE BELOW
# - **TEMP & DEWP = TEMPERATURE & DEW POINT IN FAHRENHEIT**
# - SLP = SEA LEVEL PRESSURE IN MILLIBARS TO NEAREST TENTH 
# - ALT = ALTIMETER SETTING IN INCHES TO NEAREST HUNDREDTH
# - STP = STATION PRESSURE IN MILLIBARS TO NEAREST TENTH
# - MAX = MAXIMUM TEMPERATURE IN FAHRENHEIT (TIME PERIOD VARIES)
# - MIN = MINIMUM TEMPERATURE IN FAHRENHEIT (TIME PERIOD VARIES)
# - PCP01 = 1-HOUR LIQUID PRECIP REPORT IN INCHES AND HUNDREDTHS
# - PCP06 = 6-HOUR LIQUID PRECIP REPORT IN INCHES AND HUNDREDTHS
# - PCP24 = 24-HOUR LIQUID PRECIP REPORT IN INCHES AND HUNDREDTHS
# - PCPXX = LIQUID PRECIP REPORT IN INCHES AND HUNDREDTHS
# - SD = SNOW DEPTH IN INCHES  

# In[23]:


df = df.drop(['GUS', 'CLG', 'SKC', 'L', 'M', 'H', 'VSB', 'MW', 'MW_1', 'MW_2', 'MW_3', 'AW', 'AW_4', 'AW_5', 'AW_6', 'W', 'SLP', 'ALT', 'STP', 'MAX', 'MIN', 'PCP01', 'PCP06', 'PCP24', 'PCPXX', 'SD'], axis=1)
df.columns


# ## Value vs AQI
# 
# There are two columns showing PM2.5, Value and AQI.Take a look at the first 20 rows of the dataframe.

# In[24]:


df.head(20)


# It appears that there is a lag between when the Value goes up or down and the corresponding AQI is changed. This may be because the AQI is calculated in a moving window or at the end of the stated period. Let's visualize these two columns to better understand.

# In[25]:


def value_aqi(date):
    fig, ax1 = plt.subplots()
    x = date.Hour
    s1 = date.Value
    ax1.plot(x, s1, 'b-')
    ax1.set_xlabel('hour')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Value', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    s2 = date.AQI
    ax2.plot(x, s2, 'r.')
    ax2.set_ylabel('AQI', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    return plt.show();


# In[26]:


value_aqi(month10[:24])


# In[27]:


value_aqi(month6[:24])


# In[28]:


value_aqi(month1[:24])


# **Possible reasons for AQI and Value not always being colinear**
# - The AQI as calculated by the data source is a moving average that smoothed over time
# - The first month of the dataset appears to be most impacted by this. It is possible the pollution station had some maintenance issues and the data is not reliable for some reason.
# 
# At some times AQI and Value track very closely as would be expected. However at others (in month 10 of 2015 for example) there is a disconnect. As we only need one feature, and the Value field is labeled with a specific unit, we will drop AQI in favor of PM2.5 Value.

# In[29]:


df = df.drop(['AQI'], axis=1)
df.columns


# ## Drop extra date/time columns
# 
# Earlier we determined that hour of day and month were both possibly useful predictors, and as such we will keep them in our data. However Year, Day, Date Key aren't needed. We will turn the Date (LST) feature into a datetime feature later.

# In[30]:


df = df.drop(['Year', 'Day', 'Date Key'], axis=1)
df.columns


# ## Create canonical date feature
# 
# As there are several date features, we can reduce these to one canonical one that is in a format that is easily parseable by Python or other programs. Also let's rename the 'Date (LST)' column to simply Date, as spaces in columns aren't ideal.

# In[31]:


from datetime import datetime

df['Date (LST)'] = pd.to_datetime(df['Date (LST)'])
df = df.rename(columns={"Date (LST)": "Date"})


# Incidentally now that we have a properly formatted Date field we can plot a time series of PM2.5 values over the entire dataset. Considering the length of time this may be messy, but let's give it a go.

# In[32]:


def time_series(start, end):
    time_series_df = df[['Date', 'Value']][(df['Date'] >= start) & (df['Date'] <= end)]
    x = time_series_df.Date
    y = time_series_df.Value
    plt.plot(x,y)
    plt.xlabel('Time')
    plt.ylabel('PM2.5 Value')
    plt.title('PM2.5 Time Series')
    return plt.show();


# In[33]:


time_series('2015','2018')


# The following graph shows a good example of outliers. These would appear to be errors in the data. In the graph above they appear in roughly the middle of the graph.

# In[34]:


time_series('2016-09-04','2016-09-07')


# It is quite clear that PM2.5 has a seasonal component. However it is also clear that there are outliers that are outside the normal trend. We can deal with these in the next section.

# ## Handling outliers
# Let's handle those outlier points. First lets take a look at the date range from the graph above.

# In[35]:


# Select the time frame from the graph above
df[['Date', 'Value']][(df['Date'] >= '2016-09-05 01') & (df['Date'] <= '2016-09-06 04')]


# It appears the two outlier values are both .985. Could this be a trend?

# In[36]:


df[['Date', 'Value']][(df['Value'] == .985)]


# **Determinations**
# - There are only 5 values that equal .985, so it does not appear to be an anomaly across the data set. 
# - However they are all in month 9 of 2016. 
# - Month 9 is a quite low pollution month, and an exploration of the data supports this. 
# 
# Considering these things it is prudent to remove these outliers.

# In[37]:


# Check shape of dataframe to ensure that rows are dropped later
df.shape


# In[38]:


df = df[df.Value != .985]
# Check shape again to confirm
df.shape


# Next up is the early part of the data set. Between 2015-09 and around the middle of 2015-10 the data appears to be unreliable. The data in this period does not follow what would be expected of pollution levels. This is the time when the pollution monitoring station was installed, so it is logical to belive that the station could have some calibration or other maintenance during this initial phase.

# In[39]:


time_series('2015-10','2016-2')


# In[40]:


time_series('2015','2015-10-25')


# In[41]:


time_series('2015','2015-10-20')


# In[42]:


# Select the time frame from the graph above
df[['Date', 'Value']][(df['Date'] >= '2015') & (df['Date'] <= '2015-10-20')]


# In[43]:


print("Shape before: ", df.shape)
df = df[df.Date > '2015-10-20 01']
# Check shape again to confirm
print("Shape after: ", df.shape)


# In[44]:


time_series('2015-10-22','2015-10-22 23')


# In[45]:


# Select the time frame from the graph above
df[['Date', 'Value']][(df['Date'] >= '2015-10-22 16') & (df['Date'] <= '2015-10-22 18')]


# In[46]:


df[['Date', 'Value']][(df['Value'] == .995)]


# In[47]:


print("Shape before: ", df.shape)
df = df[df.Value != .995]
print("Shape after: ", df.shape)


# After removing these outliers let's take a look at the full time series plot one more time.

# In[48]:


time_series('2015','2018')


# ## Handle NaNs

# As stated at the beginning, there are quite a few NaN values in our dataset. Most of them have been taken out by dropping columns as the majority of some of the weather features had null values.
# 
# Process for handling NaNs
# 1. Determine where NaNs exist
# 2. Decide on a per feature (column) basis whether to drop NaN records (rows)
# 3. Decide on a per feature (column) basis if/how to interpolate data for NaN records. 
#     - Interpolating can be either a mean of previous and next values, a constant number, or some other method. 

# In[49]:


# Are there null values in our dataset?
df.isnull().values.any()


# In[50]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)]


# Each feature will be handled independently. Some thoughts on each (Date, Month, Hour, Value, DIR, SPD, TEMP, DEWP):
# 
# 1. **Date, Month,** and **Hour** should have no NaNs, as this was how the two data sets (AQI and weather) were merged. However we should check to be sure.
# 2. Since we are predicting for the **Value** feature, any record with a NaN for Value should be removed.
# 3. Currently less is known about **DIR, SPD, TEMP, DEWP**. Let's explore more and see.

# In[51]:


# 1. Check if Date, Month, or Hour have NaNs
print("Date contains nulls:", df.Date.isnull().values.any())
print("Month contains nulls:", df.Month.isnull().values.any())
print("Hour contains nulls:", df.Hour.isnull().values.any())


# In[52]:


df.shape


# In[53]:


# 2. Drop any row where Value is NaN

# Show rows where value is NaN
df[df['Value'].isnull()]


# There are a total of 772 rows with NaNs in the Value column out of 14687. We can drop these.

# In[54]:


df = df.dropna(axis=0,subset=['Value'])
df.shape


# For the next 4 features let's explore these a bit to see if there is a pattern to the NaNs so that they can be intelligently replaced.

# In[55]:


# Show rows where DIR is NaN
df[df['DIR'].isnull()]


# In[56]:


# Show rows where SPD is NaN
df[df['SPD'].isnull()]


# In[57]:


# Show rows where TEMP is NaN
df[df['TEMP'].isnull()]


# In[58]:


# Show rows where DEWP is NaN
df[df['DEWP'].isnull()]


# In[59]:


# Show rows where DEWP & TEMP is NaN
df[df['DEWP'].isnull() & df['TEMP'].isnull()]


# In[60]:


# Show rows where DEWP & TEMP & DIR is NaN
df[df['DEWP'].isnull() & df['TEMP'].isnull() & df['DIR'].isnull()]


# Considering the very small number of rows in DEWP and TEMP that have NaNs we can safely drop them. We will only lose 24 records from the data.

# In[61]:


df = df.dropna(axis=0,subset=['DEWP', 'TEMP'])
df.shape


# In[62]:


df[df.DIR == 990]


# The last two features to contend with are SPD and DIR. Both have a high number of NaNs. We have several options:
# 
# - A constant value (0 for example)
# - A value from another record. For example the previous record or the next record that is not NaN
# - A mean, median, mode
# - A value determined from another model
# - Drop the records with NaNs
# 
# The most straightforward approach would be to simply drop all NaN's. However it is possible that removing them could take away valuable signal for the machine learning model. Choosing the right method may require trial and error. For now we can attempt option 2, imputing values from other records. 
# 
# Consider taking the mean wind direction and imputing that value. This will allow us to keep the rows with missing data, but it will also give a logical value for these rows. Using the mean value would ignore the cyclical nature of weather and may give bad signal to the machine learning algorithm. Using a constant value is a poor idea because it imputes signal where none may exist. For example 0 for DIR would mean wind coming from precisely North, which while a common wind direction for Ulaanbaatar may not be accurate for that time.
# 
# Assuming that previous hours wind speed and dir values will have an impact on future pollution values using the previous value would seem to be logical. 

# In[63]:


df.describe()


# Out of 13891 rows we have 1258 missing rows for SPD and 2849 missing rows for DIR. Next we will impute previous values to SPD and DIR.

# In[64]:


df = df.fillna(method='ffill')


# Using the forward fill method of filling built into Pandas is very convenient. Let's check to make sure we have all of the rows filled properly.

# In[65]:


df.describe()


# It turns out we have one missing value for DIR. It happens to be the first row in our dataframe so the forward fill can't fill it. Let's drop the row.

# In[66]:


df = df.dropna(axis=0,subset=['DIR'])


# Now that we have handled all NaNs and we won't be removing any more data we can recreate the index.

# In[67]:


df = df.reset_index(drop=True)


# ## Handle 990 DIR Values

# The DIR feature has many values of 990. These are obviously outside the 360 arc that makes a circle. The metadata for the weather data source (NOAA) says these 990 values indicate a variable wind direction. 

# In[68]:


df[df.DIR == 990]


# More than half of our dataset has 990 in the DIR column. This is a challenge considering we aren't yet sure how wind speed impacts PM2.5 levels. Let's explore and see if we can determine an illeligent way to handle these values.

# In[69]:


x = df.DIR
y = df.Value
plt.scatter(x,y)
plt.xlabel('Wind Direction')
plt.ylabel('PM2.5')
plt.title('Value and DIR')
plt.xlim(0, 990)
plt.show()


# In[70]:


plt.hist(df.DIR[(df.DIR < 990) & (df.Value > .2)], 50)
plt.xlabel('Wind Direction')
plt.ylabel('Frequency')
plt.title('Wind Direction Frequency')
plt.show()


# The value of 990 is the most varied direction in our dataset. Before we forward filled the data we only had about 5,000 990 DIR values, and now we have over 7,000. While this could be an issue, in the interest of preserving as much of the data as possible let's keep these rows for now. 
# 
# Direction is a particularly tricky feature because it is a circular feature. The field of circular statistics addresses these sorts of issues. However taking a more direct approach may be the best for a first try. In the feature engineering section we can handle these values by converting the DIR feature to a categorical variable using the cardinal directions. 

# # Feature Engineering

# ## Convert cyclical & circular features
# 
# 

# ### DIR
# Previously we found that more than half our data set contains 990 values in the DIR column. We won't be able to convert this feature to a circular feature (0-360 degrees) given these values. Alternatively let's change this feature to a categorical one. We can split the data set equally into cardinal directions as so:
# 
# Source: http://snowfence.umn.edu/Components/winddirectionanddegreeswithouttable3.htm
# 
# ![image.png](attachment:image.png)
# 
# For the 990 values we can simply assign them the value "V" for variable. Let's implement this

# In[71]:


pd.unique(df.DIR)


# We will create a dictionary of unique direction values and assign them to a cardinal direction. We will assign 990 to "V"

# In[72]:


cardinal_directions = {"DIR": {990: "V", 190: "S", 80: "E", 240: "WSW", 220: "SW", 
160: "SSE", 170: "S", 250: "WSW", 290: "WNW", 330: "NNW", 
                    340: "NNW", 320: "NW", 180: "S", 130: "SE", 140: "SE", 70: "ENE", 
                                            30: "NNE", 40: "NE", 150: "SSE", 200: "SSW", 360: "N", 50: "NE", 280: "W", 310: "NW", 300: "WNW", 350: "N", 270: "W", 120: "ESE", 260: "W", 90: "E", 60: "ENE", 100: "E", 210: "SSW", 10: "N", 230: "SW", 110: "ESE", 20: "NNE"}}


# Let's replace the number values with their corresponding cardinal directions.

# In[73]:


df.replace(cardinal_directions, inplace=True)
df.dtypes


# However when we get to training our models we will run into an issue. Regression models require a numerical input and numerical output. Considering that we now have cardinal directions this is an issue. One solution to this problem is to again convert these categories into numerical features. 
# 
# We could simply assign a number to each cardinal direction, but this would again create the issue of having numbers that are larger not being properly represented as being close to the smaller number. Helpfully Pandas has a great method called get_dummies that will allow us to create a new feature for each direction. Then that record will receive a 0 or 1 whether it is that direction or not. The benefit of this is to eliminate the problem of this circular feature being represented linearly and to give the machine learning models a number to use when estimating.

# In[74]:


df.head(10)


# In[75]:


df = pd.concat([df, pd.get_dummies(df.DIR)], axis=1)


# In[76]:


df.columns


# In[77]:


df = df.drop(['DIR'], axis=1)


# In[78]:


df.dtypes


# ### Hour and Month - Convert cyclical feature

# Hour runs from 0-24 and Month runs from 0-12. A machine learning algorithm would not properly identify that hour 0 and 24 are next to each other. It also would not identify that month 0 and 12 are next to each other. In order to provide a better signal we can convert these features. 
# 
# The code and math behind this method comes from David Kaleko's blog on handling cyclical features. Similar methods have been found in other places but this is a great straightforward explanation showing the impact on model performance.
# Link: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

# In[79]:


df['hr_sin'] = np.sin(df.Hour*(2.*np.pi/24))
df['hr_cos'] = np.cos(df.Hour*(2.*np.pi/24))
df['month_sin'] = np.sin((df.Month-1)*(2.*np.pi/12))
df['month_cos'] = np.cos((df.Month-1)*(2.*np.pi/12))


# Visualizing the features we created we can see they are now circular instead of linear. 

# In[80]:


df.plot.scatter('hr_sin','hr_cos').set_aspect('equal')
df.plot.scatter('month_sin','month_cos').set_aspect('equal')
plt.show()


# We no longer need the Month and Hour columns so we can safely drop them.

# In[81]:


df = df.drop(['Month', 'Hour'], axis=1)


# In[82]:


df.head(10)


# ## Convert Value Field from mg to µg

# The US Embassy stores their PM2.5 values in mg/m^3. However when converting to the US EPA AQI standard the measurement used is µg/m^3. This is easy enough to convert by multiplying by 1,000. The purpose of this is to make it easy to display the results of the model and calculate AQI without any further computation.

# In[83]:


# 1 mg = 1,000 µg
df['Value'] = df.Value * 1000


# In[84]:


df.head(5)


# ## Convert TEMP and DEWP from F to C

# The source of the weather data comes from NOAA, which stores temperatures and dew points in Fahrenheit. As our end user is more familiar with Celsius we will convert these features now to eliminate the need to when displaying to the user.

# In[85]:


# Formula to convert F to C is: [°C] = ([°F] - 32) × 5/9
df['TEMP'] = (df.TEMP - 32) * 5.0/9.0


# In[86]:


# Formula to convert F to C is: [°C] = ([°F] - 32) × 5/9
df['DEWP'] = (df.DEWP - 32) * 5.0/9.0


# In[87]:


df.head(5)


# ## Future Actions for Feature Engineering
# 
# There are many possible feature engineering avenues we could take. Here are a few with some explanation. Depending on the results from our model we may need to visit these. 
# 
# ### Previous hour value feature. 
# In Ulaanbaatar the majority of the pollution is created in the ger district, which surrounds the city on all sides except the south. As wind blows the pollution is pushed through the city. Considering that the pollution could take several hours to clear the city it may be helpful to create features to show the previous hour(s) wind speed, direction, dew point, and pollution value. The benefits are currently unclear but we can experiment with this later.
# 
# ### Moving average feature
# Our dataset currently exists as a hourly measurements. As we stated previously there could be some benefit to creating features of previous hours values. One such feature could be a moving average of the previous several hours or day. Again it is unclear of the benefits currently.
# 
# 
# ### Number of days since start
# We currently have a date feature that we will remove before we train our model. However it could be useful to create a feature of number of days since the start of the dataset. This could have some trade offs as we would need to keep track of this going forward with retraining data but may give good signal. 

# ## Get final dataset

# Before we move to the training and evaluating the model we can finalize our dataframe. The only unneeded feature we have remaining is the Date feature. 

# In[88]:


df.head(5)


# In[89]:


df = df.drop(['Date'], axis=1)
df.columns


# In[90]:


df.dtypes


# # Split into Training and Test Data

# Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

# ## Prepare Data Set for Regression

# The first models we will attempt will be regression models. We will prepare the our data frame by separating it into independent variables (X) and dependent variable being predicted (y).

# In[91]:


y = df['Value']
X = df.drop(['Value'], axis=1)


# In[92]:


X.head()


# In[93]:


y.head()


# In[94]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)


# In[95]:


X_train.shape, y_train.shape


# In[96]:


X_test.shape, y_test.shape


# ## Prepare Data Set for Classification

# Our predicted feature (y) is currently a continuous feature. This is what is required for the regression algorithms. However we also would like to apply classification algorithms to our data. This will require us to modify our Value (y) feature.

# In[110]:


y.head()


# Classification algorithms predict a category as an output instead of a continuous variable. This works quite well in the case of air pollution as the EPA has existing categories for classifying AQI. 
# 
# ![image.png](attachment:image.png)
# 
# Source: https://airnow.gov/index.cfm?action=aqibasics.aqi
# 
# This is an easy to understand system for end users and also a good standard by which to base our classifier on. A few notes about the EPA AQI standard.
# 
# - The scale does not go above 500. As Ulaanbaatar PM2.5 pollution in fact does go above 500 regularly during winter months, we will need to handle this in some way.
# - The PM2.5 measurement and AQI are different numbers. The measurement is placed against a scale that is used to identify the AQI.
# - AQI is a general scale that can combine multiple measures of air pollution. However in this case we will only be using PM2.5.

# The official breakpoints between the categories has changed over time. We will be using the official 

# In[ ]:


from sklearn.model_selection import train_test_split

X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X, y_cl, test_size=.3, random_state=0)


# # Implement ML Algorithms

# There are many algorithms to choose from in scikit learn. However the goal of this project is to implement the final model in Azure ML. This gives us a subset of options to work with. We will first implement these algorithms using scikit learn for illustration purposes and then implement the best few in Azure ML for creating the final web service. 
# 
# Algorithms to test:
# 
# **Regression**
# - Linear regression
# - Neural network regression
# - Decision forest
# - Boosted decision tree
# 
# **Classification**
# - Multiclass neural network
# - Multiclass logistic regression
# - Multiclass decision forest
# - Multiclass decision jungle

# ## Regression

# ### Linear Regression

# In[97]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create linear regression object
regr = linear_model.LinearRegression()


# In[98]:


# Train the model using the training sets
regr.fit(X_train, y_train)


# In[99]:


# Make predictions using the testing set
lin_pred = regr.predict(X_test)


# In[100]:


regr.score(X_test, y_test)


# In[101]:


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, lin_pred))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lin_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lin_pred))


# In[102]:


plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()


# **Conclusion: Linear Regression**
# 
# Clearly this model does not perform well. The absolute error is quite high and the r-squared is only 0.36. 

# ### Neural Network Regression

# In[103]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()


# In[104]:


# Train the model using the training sets
mlp.fit(X_train, y_train)


# In[105]:


# Score the model
mlp.score(X_test, y_test)


# In[106]:


# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)


# In[107]:


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, nnr_pred))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, nnr_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, nnr_pred))


# In[108]:


plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()


# **Neural Network Conclusion**
# 
# The neural network fared better. Our r-squared value is up to .49. This still is quite poor though and says that our model only explains about half of our data well.

# ### Decision Forest

# In[109]:


from sklearn import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=100)


# In[ ]:


# Train the model using the training sets
regr_rf.fit(X_train, y_train)


# In[ ]:


# Score the model
regr_rf.score(X_test, y_test)


# In[ ]:


# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)


# In[ ]:


plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()


# ### Decision Tree + AdaBoost

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Decision Tree Regressor object
tree_1 = DecisionTreeRegressor()

tree_2 = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=600, learning_rate=.1)


# In[ ]:


# Train the model using the training sets
tree_1.fit(X_train, y_train)
tree_2.fit(X_train, y_train)


# In[ ]:


# Score the decision tree model
tree_1.score(X_test, y_test)


# In[ ]:


# Score the boosted decision tree model
tree_2.score(X_test, y_test)


# In[ ]:


# Make predictions using the testing set
tree_1_pred = tree_1.predict(X_test)
tree_2_pred = tree_2.predict(X_test)


# In[ ]:


plt.scatter(y_test, tree_1_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Tree Predicted vs Actual')
plt.show()


# In[ ]:


plt.scatter(y_test, tree_2_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Boosted Decision Tree Predicted vs Actual')
plt.show()


# ## Classification

# ### Multiclass neural network

# ### Multiclass logistic regression

# ### Multiclass decision forest

# ### Multiclass decision jungle

# # Evaluate Models

# Let's compare all of our models scores. The regression and classification models are scored differently, with regression using r-squared and the classification models using an accuracy score. We can at the very least compare models of the same type.

# In[ ]:


print("Regression models:")
print("Linear regression score: ", regr.score(X_test, y_test))
print("Neural network regression score: ", mlp.score(X_test, y_test))
print("Decision forest score: ", regr_rf.score(X_test, y_test))
print("Boosted decision tree score: ", tree_2.score(X_test, y_test))
print(" ")
print("Classification models:")
print("Multiclass neural network score: ")
print("Multiclass logitic regression score: ")
print("Multiclass decision forest score: ")
print("Multiclass decision jungle score: ")


# # To Do 
# 2. Predict specific value or should we predict category?
# 4. Separate into training and test data
# 5. Evaluate which model will best fit data.
# 6. Redo visualizations to be cleaner and give explanations
