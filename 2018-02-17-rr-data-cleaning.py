
# coding: utf-8

# # Ulaanbaatar PM2.5 Prediction
# 
# The purpose of this notebook is to create a predictive model of PM2.5 levels in Ulaanbaatar, Mongolia. 

# 
# # To Do 
# 1. Make a column that is number of days since start.
# 2. Predict specific value or should we predict category?
# 3. Convert hours and wind direction to something more understandable by the machine learning algorithm.
# 4. Separate into training and test data
# 5. Evaluate which model will best fit data.
# 6. Redo visualizations to be cleaner and give explanations

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

# In[1]:


# Import relevant items
import pandas as pd
import numpy as np


# In[2]:


# Let's first load the data and take a look at what we have.
df = pd.read_csv('weather-and-aqi-v4.csv')


# The head of the dataframe shows lots of columns and LOTS of NaN's.

# In[3]:


print(df.head())
print(df.columns)


# In[4]:


df.dtypes


# There are a large number of columns that are unneccesary. There are duplicate columns for Date, Year, Month, Day, and Hour. There are also columns for location name, station id, units, and intervals of measurement that are not useful in analysis. For now we can leave these columns in. 

# In[5]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [16,9]


# # Visualizing Features<a name="visualize"></a>
# 
# **Let's plot the various features (pollution level, time, month, wind speed, etc) to find any relationships.**
# 
# When plotting the PM2.5 concentration by month you can clearly see that winter months have a much larger variation in pollution levels, including some very high levels.

# In[6]:


x = df['Month']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('Month')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Month')
plt.show()


# Looking at the value plot by hour you can see there are two spikes each day, one between 9-11AM and the other starting around 20 and continuing through the night until 4.

# In[7]:


x = df['Hour']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('Hour')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Hour')
plt.show()


# At higher windspeeds you notice a big drop in the recorded PM2.5 levels. It seems wind speed may be a good feature to predict PM2.5.

# In[8]:


x = df['SPD']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('SPD')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Windspeed')
plt.show()


# Make facet or subplot of PM2.5 levels by windspeed with one subplot per month. This will show if these lower measured values at low windspeeds are evenly distrbuted throughout the year or if they are mostly in certain months.

# In[9]:


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


# In[10]:


x = month1['SPD']
y = month1['AQI']
plt.scatter(x,y, color='blue')
plt.xlabel('Wind Speed')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Windspeed - January')
plt.show()


# In[11]:


x = month1['TEMP']
y = month1['Value']
plt.scatter(x,y)
plt.xlabel('Temperature')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Temperature - January')
plt.show()


# In[12]:


x = month1['TEMP']
y = month1['SPD']
col = np.where(month1['AQI']<100,'None',np.where(month1['AQI']>100,'red','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Temperature')
plt.ylabel('Wind Speed')
plt.title('PM2.5 > 100 by Temperature and Windspeed - January')


# In[13]:


plt.show()


# In[14]:


x = month1['TEMP']
y = month1['SPD']
col = np.where(month1['AQI']<100,'green',np.where(month1['AQI']>100,'None','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Temperature')
plt.ylabel('Wind Speed')
plt.title('PM2.5 < 100 by Temperature and Windspeed - January')
plt.show()


# In[15]:


x = month1['Hour']
y = month1['TEMP']
col = np.where(month1['AQI']<100,'green',np.where(month1['AQI']>100,'None','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Hour')
plt.ylabel('Temperature')
plt.title('PM2.5 < 100 by Temperature and Hour - January')
plt.show()


# In[16]:


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

# In[17]:


df.columns


# Source.Name, Site, Parameter, Unit, Duration, USAF, and WBAN are the same for every row, and as such are not useful features. These are site identifiers for the weather station, the duration of measurement (which is constant), and the unit of PM2.5 measurement (which is in milligrams per cubic meter).

# In[18]:


df = df.drop(['Source.Name', 'Site', 'Parameter', 'Unit', 'Duration', 'USAF', 'WBAN'], axis=1)


# Date Key.1, Year.1, Month.1, Day.1, and Hour.1 are duplicates of the original date features. These were used to create the date key that was then used to combine the PM2.5 and weather data sets. Removing these will cause no harm.

# In[19]:


df = df.drop(['Date Key.1', 'Year.1', 'Month.1', 'Day.1', 'Hour.1'], axis=1)


# In[20]:


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

# In[21]:


df = df.drop(['GUS', 'CLG', 'SKC', 'L', 'M', 'H', 'VSB', 'MW', 'MW_1', 'MW_2', 'MW_3', 'AW', 'AW_4', 'AW_5', 'AW_6', 'W', 'SLP', 'ALT', 'STP', 'MAX', 'MIN', 'PCP01', 'PCP06', 'PCP24', 'PCPXX', 'SD'], axis=1)
df.columns


# ## Value vs AQI
# 
# There are two columns showing PM2.5, Value and AQI.Take a look at the first 20 rows of the dataframe.

# In[22]:


df.head(20)


# It appears that there is a lag between when the Value goes up or down and the corresponding AQI is changed. This may be because the AQI is calculated in a moving window or at the end of the stated period. Let's visualize these two columns to better understand.

# In[23]:


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


# In[24]:


value_aqi(month10[:24])


# In[25]:


value_aqi(month6[:24])


# In[26]:


value_aqi(month1[:24])


# **Possible reasons for AQI and Value not always being colinear**
# - The AQI as calculated by the data source is a moving average that smoothed over time
# - The first month of the dataset appears to be most impacted by this. It is possible the pollution station had some maintenance issues and the data is not reliable for some reason.
# 
# At some times AQI and Value track very closely as would be expected. However at others (in month 10 of 2015 for example) there is a disconnect. As we only need one feature, and the Value field is labeled with a specific unit, we will drop AQI in favor of PM2.5 Value.

# In[27]:


df = df.drop(['AQI'], axis=1)
df.columns


# ## Drop extra date/time columns
# 
# Earlier we determined that hour of day and month were both possibly useful predictors, and as such we will keep them in our data. However Year, Day, Date Key aren't needed. We will turn the Date (LST) feature into a datetime feature later.

# In[28]:


df = df.drop(['Year', 'Day', 'Date Key'], axis=1)
df.columns


# ## Create canonical date feature
# 
# As there are several date features, we can reduce these to one canonical one that is in a format that is easily parseable by Python or other programs. Also let's rename the 'Date (LST)' column to simply Date, as spaces in columns aren't ideal.

# In[29]:


from datetime import datetime

df['Date (LST)'] = pd.to_datetime(df['Date (LST)'])
df = df.rename(columns={"Date (LST)": "Date"})


# Incidentally now that we have a properly formatted Date field we can plot a time series of PM2.5 values over the entire dataset. Considering the length of time this may be messy, but let's give it a go.

# In[30]:


def time_series(start, end):
    time_series_df = df[['Date', 'Value']][(df['Date'] >= start) & (df['Date'] <= end)]
    x = time_series_df.Date
    y = time_series_df.Value
    plt.plot(x,y)
    plt.xlabel('Time')
    plt.ylabel('PM2.5 Value')
    plt.title('PM2.5 Time Series')
    return plt.show();


# In[31]:


time_series('2015','2018')


# The following graph shows a good example of outliers. These would appear to be errors in the data. In the graph above they appear in roughly the middle of the graph.

# In[32]:


time_series('2016-09-04','2016-09-07')


# It is quite clear that PM2.5 has a seasonal component. However it is also clear that there are outliers that are outside the normal trend. We can deal with these in the next section.

# ## Handling outliers
# Let's handle those outlier points. First lets take a look at the date range from the graph above.

# In[33]:


# Select the time frame from the graph above
df[['Date', 'Value']][(df['Date'] >= '2016-09-05 01') & (df['Date'] <= '2016-09-06 04')]


# It appears the two outlier values are both .985. Could this be a trend?

# In[34]:


df[['Date', 'Value']][(df['Value'] == .985)]


# **Determinations**
# - There are only 5 values that equal .985, so it does not appear to be an anomaly across the data set. 
# - However they are all in month 9 of 2016. 
# - Month 9 is a quite low pollution month, and an exploration of the data supports this. 
# 
# Considering these things it is prudent to remove these outliers.

# In[35]:


# Check shape of dataframe to ensure that rows are dropped later
df.shape


# In[36]:


df = df[df.Value != .985]
# Check shape again to confirm
df.shape


# Next up is the early part of the data set. Between 2015-09 and around the middle of 2015-10 the data appears to be unreliable. The data in this period does not follow what would be expected of pollution levels. This is the time when the pollution monitoring station was installed, so it is logical to belive that the station could have some calibration or other maintenance during this initial phase.

# In[37]:


time_series('2015-10','2016-2')


# In[38]:


time_series('2015','2015-10-25')


# In[39]:


time_series('2015','2015-10-20')


# In[40]:


# Select the time frame from the graph above
df[['Date', 'Value']][(df['Date'] >= '2015') & (df['Date'] <= '2015-10-20')]


# In[41]:


print("Shape before: ", df.shape)
df = df[df.Date > '2015-10-20 01']
# Check shape again to confirm
print("Shape after: ", df.shape)


# In[42]:


time_series('2015-10-22','2015-10-22 23')


# In[43]:


# Select the time frame from the graph above
df[['Date', 'Value']][(df['Date'] >= '2015-10-22 16') & (df['Date'] <= '2015-10-22 18')]


# In[44]:


df[['Date', 'Value']][(df['Value'] == .995)]


# In[45]:


print("Shape before: ", df.shape)
df = df[df.Value != .995]
print("Shape after: ", df.shape)


# After removing these outliers let's take a look at the full time series plot one more time.

# In[46]:


time_series('2015','2018')


# ## Handle NaNs

# As stated at the beginning, there are quite a few NaN values in our dataset. Most of them have been taken out by dropping columns as the majority of some of the weather features had null values.
# 
# Process for handling NaNs
# 1. Determine where NaNs exist
# 2. Decide on a per feature (column) basis whether to drop NaN records (rows)
# 3. Decide on a per feature (column) basis if/how to interpolate data for NaN records. 
#     - Interpolating can be either a mean of previous and next values, a constant number, or some other method. 

# In[47]:


# Are there null values in our dataset?
df.isnull().values.any()


# In[48]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)]


# Each feature will be handled independently. Some thoughts on each (Date, Month, Hour, Value, DIR, SPD, TEMP, DEWP):
# 
# 1. **Date, Month,** and **Hour** should have no NaNs, as this was how the two data sets (AQI and weather) were merged. However we should check to be sure.
# 2. Since we are predicting for the **Value** feature, any record with a NaN for Value should be removed.
# 3. Currently less is known about **DIR, SPD, TEMP, DEWP**. Let's explore more and see.

# In[49]:


# 1. Check if Date, Month, or Hour have NaNs
print("Date contains nulls:", df.Date.isnull().values.any())
print("Month contains nulls:", df.Month.isnull().values.any())
print("Hour contains nulls:", df.Hour.isnull().values.any())


# In[50]:


df.shape


# In[51]:


# 2. Drop any row where Value is NaN

# Show rows where value is NaN
df[df['Value'].isnull()]


# There are a total of 772 rows with NaNs in the Value column out of 14687. We can drop these.

# In[52]:


df = df.dropna(axis=0,subset=['Value'])
df.shape


# For the next 4 features let's explore these a bit to see if there is a pattern to the NaNs so that they can be intelligently replaced.

# In[53]:


# Show rows where DIR is NaN
df[df['DIR'].isnull()]


# In[54]:


# Show rows where SPD is NaN
df[df['SPD'].isnull()]


# In[55]:


# Show rows where TEMP is NaN
df[df['TEMP'].isnull()]


# In[56]:


# Show rows where DEWP is NaN
df[df['DEWP'].isnull()]


# In[57]:


# Show rows where DEWP & TEMP is NaN
df[df['DEWP'].isnull() & df['TEMP'].isnull()]


# In[58]:


# Show rows where DEWP & TEMP & DIR is NaN
df[df['DEWP'].isnull() & df['TEMP'].isnull() & df['DIR'].isnull()]


# Considering the very small number of rows in DEWP and TEMP that have NaNs we can safely drop them. We will only lose 24 records from the data.

# In[59]:


df = df.dropna(axis=0,subset=['DEWP', 'TEMP'])
df.shape


# The last two features to contend with are SPD and DIR. Both have a high number of NaNs. We have several options:
# 
# - A constant value (0 for example)
# - A value from another record. For example the previous record or the next record that is not NaN
# - A mean, median, mode
# - A value determined from another model
# - Drop the records with NaNs
# 
# Choosing the right method may require trial and error. However without testing each method iteratively let's think about the best approach. Consider taking the mean wind direction and imputing that value. 
# 
# The most straightforward approach would be to simply drop all NaN's. However it is possible that removing them could take away valuable signal for the machine learning model. So for now we can attempt option 2, imputing values from other records.

# In[60]:


df.describe()


# # Feature Engineering

# ## Convert cyclical features
# 
# 

# ## Create previous hour value feature

# ## Create moving average feature
