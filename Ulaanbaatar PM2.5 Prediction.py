
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Ulaanbaatar-PM2.5-Prediction" data-toc-modified-id="Ulaanbaatar-PM2.5-Prediction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ulaanbaatar PM2.5 Prediction</a></span></li><li><span><a href="#Importing-data-and-assumptions" data-toc-modified-id="Importing-data-and-assumptions-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Importing data and assumptions</a></span><ul class="toc-item"><li><span><a href="#Remove-unneeded-features" data-toc-modified-id="Remove-unneeded-features-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Remove unneeded features</a></span></li></ul></li><li><span><a href="#Visualizing-Features" data-toc-modified-id="Visualizing-Features-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Visualizing Features</a></span><ul class="toc-item"><li><span><a href="#Determinations-made-from-visualizations" data-toc-modified-id="Determinations-made-from-visualizations-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Determinations made from visualizations</a></span></li></ul></li><li><span><a href="#Clean-Data" data-toc-modified-id="Clean-Data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Clean Data</a></span><ul class="toc-item"><li><span><a href="#Create-canonical-date-feature" data-toc-modified-id="Create-canonical-date-feature-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Create canonical date feature</a></span></li><li><span><a href="#Create-previous-value-features" data-toc-modified-id="Create-previous-value-features-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Create previous value features</a></span></li><li><span><a href="#Handling-outliers" data-toc-modified-id="Handling-outliers-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Handling outliers</a></span></li><li><span><a href="#Handle-NaNs" data-toc-modified-id="Handle-NaNs-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Handle NaNs</a></span></li><li><span><a href="#Handle-990-DIR-Values" data-toc-modified-id="Handle-990-DIR-Values-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Handle 990 DIR Values</a></span></li></ul></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Convert-cyclical-&amp;-circular-features" data-toc-modified-id="Convert-cyclical-&amp;-circular-features-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Convert cyclical &amp; circular features</a></span></li><li><span><a href="#Convert-Value-Field-from-mg-to-µg" data-toc-modified-id="Convert-Value-Field-from-mg-to-µg-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Convert Value Field from mg to µg</a></span></li><li><span><a href="#Convert-TEMP-and-DEWP-from-F-to-C" data-toc-modified-id="Convert-TEMP-and-DEWP-from-F-to-C-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Convert TEMP and DEWP from F to C</a></span></li><li><span><a href="#Convert-SPD-from-Mph-to-Kph" data-toc-modified-id="Convert-SPD-from-Mph-to-Kph-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Convert SPD from Mph to Kph</a></span></li><li><span><a href="#Create-day-of-the-week-feature" data-toc-modified-id="Create-day-of-the-week-feature-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Create day of the week feature</a></span></li><li><span><a href="#Previous-hour-values" data-toc-modified-id="Previous-hour-values-5.6"><span class="toc-item-num">5.6&nbsp;&nbsp;</span>Previous hour values</a></span></li><li><span><a href="#Future-Actions-for-Feature-Engineering" data-toc-modified-id="Future-Actions-for-Feature-Engineering-5.7"><span class="toc-item-num">5.7&nbsp;&nbsp;</span>Future Actions for Feature Engineering</a></span></li></ul></li><li><span><a href="#Split-into-Training-and-Test-Data" data-toc-modified-id="Split-into-Training-and-Test-Data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Split into Training and Test Data</a></span><ul class="toc-item"><li><span><a href="#Prepare-Data-Set-for-Regression" data-toc-modified-id="Prepare-Data-Set-for-Regression-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Prepare Data Set for Regression</a></span></li><li><span><a href="#Prepare-Data-Set-for-Classification" data-toc-modified-id="Prepare-Data-Set-for-Classification-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Prepare Data Set for Classification</a></span></li></ul></li><li><span><a href="#Implement-ML-Algorithms" data-toc-modified-id="Implement-ML-Algorithms-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Implement ML Algorithms</a></span><ul class="toc-item"><li><span><a href="#Regression" data-toc-modified-id="Regression-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Regression</a></span></li><li><span><a href="#Classification" data-toc-modified-id="Classification-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Classification</a></span></li></ul></li><li><span><a href="#Evaluate-Models" data-toc-modified-id="Evaluate-Models-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Evaluate Models</a></span></li><li><span><a href="#Observations" data-toc-modified-id="Observations-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Observations</a></span><ul class="toc-item"><li><span><a href="#Analysis-after-run" data-toc-modified-id="Analysis-after-run-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Analysis after run</a></span></li></ul></li></ul></div>

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

# In[90]:


# Import relevant items
import pandas as pd
import numpy as np

import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[91]:


# Let's first load the data and take a look at what we have.
df = pd.read_csv('weather-and-aqi-v4.csv')


# The head of the dataframe shows lots of columns and LOTS of NaN's.

# In[92]:


print(df.head())
print(df.columns)


# In[93]:


df.dtypes


# There are a large number of columns that are unneccesary. There are duplicate columns for Date, Year, Month, Day, and Hour. There are also columns for location name, station id, units, and intervals of measurement that are not useful in analysis. For now we can leave these columns in. 

# In[94]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [16,9]


# ## Remove unneeded features

# Thus far we have looked at the data available and relationships of some key features. Another key factor is determining what input data will be available to predict PM2.5. This model aims to predict PM2.5 levels into the future. This is possible due to weather forecasting providing inputs for the model. As such we will be limited to the data that is available from the weather forecast provider. Data that is available from several weather API forecasters:
# 
# - Temperature
# - Humidity
# - Wind speed
# - Wind direction
# - Dew point
# 
# To start, let's drop columns that clearly have no value as features.
# 
# Source.Name, Site, Parameter, Unit, Duration, USAF, and WBAN are the same for every row, and as such are not useful features. These are site identifiers for the weather station, the duration of measurement (which is constant), and the unit of PM2.5 measurement (which is in milligrams per cubic meter).
# 
# Date Key.1, Year.1, Month.1, Day.1, and Hour.1 are duplicates of the original date features. These were used to create the date key that was then used to combine the PM2.5 and weather data sets. Removing these will cause no harm.
# 
# Earlier we determined that hour of day and month were both possibly useful predictors, and as such we will keep them in our data. However Year, Day, Date Key aren't needed. We will turn the Date (LST) feature into a datetime feature later.
# 
# As we only need one measure of air pollution, and the Value field is labeled with a specific unit, we will drop AQI in favor of PM2.5 Value.
# 
# The remaining fields are either 1) date fields, 2) PM2.5 values or their derived AQI values, and 3) weather data. As noted before we will only have a few features available for use in prediction. As such we will remove the rest. Below is an inventory of the available weather features. Those marked in bold will be kept, all others will be removed.
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

# In[95]:


#drop unneeded features
df = df.drop(['Year', 'Day', 'Date Key', 'Date Key.1', 'Year.1', 'Month.1', 'Day.1', 'Hour.1', 'AQI', 
              'Source.Name', 'Site', 'Parameter', 'Unit', 'Duration', 'USAF', 'WBAN', 'GUS', 'CLG', 'SKC', 'L', 
              'M', 'H', 'VSB', 'MW', 'MW_1', 'MW_2', 'MW_3', 'AW', 'AW_4', 'AW_5', 'AW_6', 'W', 'SLP', 
              'ALT', 'STP', 'MAX', 'MIN', 'PCP01', 'PCP06', 'PCP24', 'PCPXX', 'SD'], axis=1)

df.columns


# # Visualizing Features
# 
# **Let's plot the various features (pollution level, time, month, wind speed, etc) to find any relationships.**
# 
# When plotting the PM2.5 concentration by month you can clearly see that winter months have a much larger variation in pollution levels, including some very high levels.

# In[96]:


x = df['Month']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('Month')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Month')
plt.show()


# Looking at the value plot by hour you can see there are two spikes each day, one between 9-11AM and the other starting around 20 and continuing through the night until 4.

# In[97]:


x = df['Hour']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('Hour')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Hour')
plt.show()


# At higher windspeeds you notice a big drop in the recorded PM2.5 levels. It seems wind speed may be a good feature to predict PM2.5.

# In[98]:


x = df['SPD']
y = df['Value']
plt.scatter(x,y)
plt.xlabel('SPD')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Windspeed')
plt.show()


# Make facet or subplot of PM2.5 levels by windspeed with one subplot per month. This will show if these lower measured values at low windspeeds are evenly distrbuted throughout the year or if they are mostly in certain months.

# In[99]:


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


# In[100]:


x = month1['SPD']
y = month1['Value']
plt.scatter(x,y, color='blue')
plt.xlabel('Wind Speed')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Windspeed - January')
plt.show()


# In[101]:


x = month1['TEMP']
y = month1['Value']
plt.scatter(x,y)
plt.xlabel('Temperature')
plt.ylabel('PM2.5 Level')
plt.title('PM2.5 by Temperature - January')
plt.show()


# In[102]:


x = month1['TEMP']
y = month1['SPD']
col = np.where(month1['Value']<100,'None',np.where(month1['Value']>100,'red','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Temperature')
plt.ylabel('Wind Speed')
plt.title('PM2.5 > 100 by Temperature and Windspeed - January')


# In[103]:


plt.show()


# In[104]:


x = month1['TEMP']
y = month1['SPD']
col = np.where(month1['Value']<100,'green',np.where(month1['Value']>100,'None','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Temperature')
plt.ylabel('Wind Speed')
plt.title('PM2.5 < 100 by Temperature and Windspeed - January')
plt.show()


# In[105]:


x = month1['Hour']
y = month1['TEMP']
col = np.where(month1['Value']<100,'green',np.where(month1['Value']>100,'None','None'))
plt.scatter(x,y, c=col)
plt.xlabel('Hour')
plt.ylabel('Temperature')
plt.title('PM2.5 < 100 by Temperature and Hour - January')
plt.show()


# In[106]:


x = month1['DIR']
y = month1['SPD']
col = np.where(month1['Value']<100,'None',np.where(month1['Value']>100,'red','None'))
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
# 
# We will visualize further after some more munging.

# # Clean Data

# ## Create canonical date feature
# 
# As there are several date features, we can reduce these to one canonical one that is in a format that is easily parseable by Python or other programs. Also let's rename the 'Date (LST)' column to simply Date, as spaces in columns aren't ideal.

# In[107]:


from datetime import datetime

df['Date (LST)'] = pd.to_datetime(df['Date (LST)'])
df = df.rename(columns={"Date (LST)": "Date"})


# Incidentally now that we have a properly formatted Date field we can plot a time series of PM2.5 values over the entire dataset. Considering the length of time this may be messy, but let's give it a go.

# In[108]:


def time_series(start, end):
    time_series_df = df[['Date', 'Value']][(df['Date'] >= start) & (df['Date'] <= end)]
    x = time_series_df.Date
    y = time_series_df.Value
    plt.plot(x,y)
    plt.xlabel('Time')
    plt.ylabel('PM2.5 Value')
    plt.title('PM2.5 Time Series')
    return plt.show();


# In[109]:


time_series('2015','2018')


# The following graph shows a good example of outliers. These would appear to be errors in the data. In the graph above they appear in roughly the middle of the graph.

# In[110]:


time_series('2016-09-04','2016-09-07')


# It is quite clear that PM2.5 has a seasonal component. However it is also clear that there are outliers that are outside the normal trend. We can deal with these in the next section.

# ## Create previous value features

# In[112]:


df.head()


# In[116]:


df['Value_1'] = df.Value.shift(periods=1)
df['TEMP_1'] = df.TEMP.shift(periods=1)
df['SPD_1'] = df.SPD.shift(periods=1)
df['DEWP_1'] = df.DEWP.shift(periods=1)


# In[118]:


df.head(10)


# ## Handling outliers
# Let's handle those outlier points in the Value feature. First lets take a look at the date range from the graph above.

# In[23]:


# Select the time frame from the graph above
df[['Date', 'Value']][(df['Date'] >= '2016-09-05 01') & (df['Date'] <= '2016-09-06 04')]


# **Determinations**
# 
# After exploring the spikes where PM2.5 values go above 500 we see that the majority of them are in the 2015-2016 winter. In 2016-2017 winter these spikes don't exist. We know from external reports that pollution is actually getting worse year over year. This leads to the conclusion that these points are possible in error and should be removed. 

# In[119]:


df = df[df.Value <= .5]


# Next up is the early part of the data set. Between 2015-09 and around the middle of 2015-10 the data appears to be unreliable. The data in this period does not follow what would be expected of pollution levels. This is the time when the pollution monitoring station was installed, so it is logical to belive that the station could have some calibration or other maintenance during this initial phase.

# In[120]:


time_series('2015-10','2016-2')


# In[121]:


time_series('2015','2015-10-20')


# In[122]:


print("Shape before: ", df.shape)
df = df[df.Date > '2015-10-20 01']
# Check shape again to confirm
print("Shape after: ", df.shape)


# In[123]:


time_series('2017-6','2017-7')


# After removing these outliers let's take a look at the full time series plot one more time.

# In[124]:


time_series('2015','2018')


# ## Handle NaNs

# As stated at the beginning, there are quite a few NaN values in our dataset. Most of them have been taken out by dropping columns as the majority of some of the weather features had null values.
# 
# Process for handling NaNs
# 1. Determine where NaNs exist
# 2. Decide on a per feature (column) basis whether to drop NaN records (rows)
# 3. Decide on a per feature (column) basis if/how to interpolate data for NaN records. 
#     - Interpolating can be either a mean of previous and next values, a constant number, or some other method. 

# In[125]:


# Are there null values in our dataset?
df.isnull().values.any()


# In[126]:


# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)]


# Each feature will be handled independently. Some thoughts on each (Date, Month, Hour, Value, DIR, SPD, TEMP, DEWP):
# 
# 1. **Date, Month,** and **Hour** should have no NaNs, as this was how the two data sets (AQI and weather) were merged. However we should check to be sure.
# 2. Since we are predicting for the **Value** feature, any record with a NaN for Value should be removed.
# 3. Currently less is known about **DIR, SPD, TEMP, DEWP**. Let's explore more and see.

# In[127]:


# 1. Check if Date, Month, or Hour have NaNs
print("Date contains nulls:", df.Date.isnull().values.any())
print("Month contains nulls:", df.Month.isnull().values.any())
print("Hour contains nulls:", df.Hour.isnull().values.any())


# In[128]:


df.shape


# In[129]:


# 2. Drop any row where Value is NaN

# Show rows where value is NaN
df[df['Value'].isnull()]


# There are a total of 772 rows with NaNs in the Value column out of 14687. We can drop these.

# In[130]:


df = df.dropna(axis=0,subset=['Value'])
df.shape


# For the next 4 features let's explore these a bit to see if there is a pattern to the NaNs so that they can be intelligently replaced.

# In[132]:


# Show rows where DIR is NaN
df[df['DIR'].isnull()]


# In[133]:


# Show rows where SPD is NaN
df[df['SPD'].isnull()]


# In[134]:


# Show rows where TEMP is NaN
df[df['TEMP'].isnull()]


# In[135]:


# Show rows where DEWP is NaN
df[df['DEWP'].isnull()]


# In[136]:


# Show rows where DEWP & TEMP is NaN
df[df['DEWP'].isnull() & df['TEMP'].isnull()]


# In[137]:


# Show rows where DEWP & TEMP & DIR is NaN
df[df['DEWP'].isnull() & df['TEMP'].isnull() & df['DIR'].isnull()]


# Considering the very small number of rows in DEWP and TEMP that have NaNs we can safely drop them. We will only lose 24 records from the data.

# In[138]:


df = df.dropna(axis=0,subset=['DEWP', 'TEMP'])
df.shape


# In[139]:


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

# In[140]:


df.describe()


# Out of 13891 rows we have 1258 missing rows for SPD and 2849 missing rows for DIR. Next we will impute previous values to SPD and DIR.

# In[141]:


df= df.fillna(method='ffill')


# Using the forward fill method of filling built into Pandas is very convenient. Let's check to make sure we have all of the rows filled properly.

# In[142]:


df.describe()


# It turns out we have one missing value for DIR. It happens to be the first row in our dataframe so the forward fill can't fill it. Let's drop the row.

# In[143]:


df = df.dropna(axis=0)


# Now that we have handled all NaNs and we won't be removing any more data we can recreate the index.

# In[144]:


df = df.reset_index(drop=True)


# In[145]:


df.dtypes


# ## Handle 990 DIR Values

# The DIR feature has many values of 990. These are obviously outside the 360 arc that makes a circle. The metadata for the weather data source (NOAA) says these 990 values indicate a variable wind direction. 

# In[146]:


df[df.DIR == 990]


# More than half of our dataset has 990 in the DIR column. This is a challenge considering we aren't yet sure how wind speed impacts PM2.5 levels. Let's explore and see if we can determine an illeligent way to handle these values.

# In[147]:


x = df.DIR
y = df.Value
plt.scatter(x,y)
plt.xlabel('Wind Direction')
plt.ylabel('PM2.5')
plt.title('Value and DIR')
plt.xlim(0, 990)
plt.show()


# In[148]:


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

# In[149]:


pd.unique(df.DIR)


# ### Convert DIR using SIN COS method

# In[150]:


df['DIR_sin'] = np.sin(df.DIR*(2.*np.pi/360))
df['DIR_cos'] = np.cos(df.DIR*(2.*np.pi/360))


# In[151]:


df.head(20)


# In[152]:


df['DIR_cos'] = df.DIR_cos.replace(np.cos(990*(2*np.pi/360)), 0)


# In[153]:


df.head(20)


# In[154]:


df['DIR_sin'][(df.DIR_sin == -1) & (df.DIR == 990)] = 0


# In[155]:


df.head(20)


# In[156]:


df.plot.scatter('DIR_sin','DIR_cos').set_aspect('equal')
plt.show()


# In[157]:


df = df.drop(['DIR'], axis=1)


# We will create a dictionary of unique direction values and assign them to a cardinal direction. We will assign 990 to "V"

# In[158]:


# cardinal_directions = {"DIR": {990: "V", 190: "S", 80: "E", 240: "WSW", 220: "SW", 
# 160: "SSE", 170: "S", 250: "WSW", 290: "WNW", 330: "NNW", 
#                     340: "NNW", 320: "NW", 180: "S", 130: "SE", 140: "SE", 70: "ENE", 
#                                             30: "NNE", 40: "NE", 150: "SSE", 200: "SSW", 360: "N", 50: "NE", 280: "W", 310: "NW", 300: "WNW", 350: "N", 270: "W", 120: "ESE", 260: "W", 90: "E", 60: "ENE", 100: "E", 210: "SSW", 10: "N", 230: "SW", 110: "ESE", 20: "NNE"}}


# Let's replace the number values with their corresponding cardinal directions.

# In[159]:


# df.replace(cardinal_directions, inplace=True)
# df.dtypes


# However when we get to training our models we will run into an issue. Regression models require a numerical input and numerical output. Considering that we now have cardinal directions this is an issue. One solution to this problem is to again convert these categories into numerical features. 
# 
# We could simply assign a number to each cardinal direction, but this would again create the issue of having numbers that are larger not being properly represented as being close to the smaller number. Helpfully Pandas has a great method called get_dummies that will allow us to create a new feature for each direction. Then that record will receive a 0 or 1 whether it is that direction or not. The benefit of this is to eliminate the problem of this circular feature being represented linearly and to give the machine learning models a number to use when estimating.

# In[160]:


df.head(10)


# In[161]:


# cardinal_index = {"DIR": {"V": 0, "SW": 6, "SSW": 2, "S": 2, "SSE": 2, "SE": 5, "ESE": 8, "E": 8, "ENE": 8, 
#                               "NE": 3, "NNE": 1, "N": 1, "NNW": 1, "NW": 4, "WNW": 7, "W": 7, "WSW": 7}}


# In[162]:


# df = pd.concat([df, pd.get_dummies(df.DIR)], axis=1)
# df.replace(cardinal_index, inplace=True)
# df.dtypes


# In[163]:


df.columns


# In[164]:


# df = df.drop(['DIR'], axis=1)
# df.dtypes


# ### Hour and Month - Convert cyclical feature

# Hour runs from 0-24 and Month runs from 0-12. A machine learning algorithm would not properly identify that hour 0 and 24 are next to each other. It also would not identify that month 0 and 12 are next to each other. In order to provide a better signal we can convert these features. 
# 
# The code and math behind this method comes from David Kaleko's blog on handling cyclical features. Similar methods have been found in other places but this is a great straightforward explanation showing the impact on model performance.
# Link: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

# In[165]:


df['hr_sin'] = np.sin(df.Hour*(2.*np.pi/24))
df['hr_cos'] = np.cos(df.Hour*(2.*np.pi/24))
df['month_sin'] = np.sin((df.Month-1)*(2.*np.pi/12))
df['month_cos'] = np.cos((df.Month-1)*(2.*np.pi/12))


# Visualizing the features we created we can see they are now circular instead of linear. 

# In[166]:


df.plot.scatter('hr_sin','hr_cos').set_aspect('equal')
df.plot.scatter('month_sin','month_cos').set_aspect('equal')
plt.show()


# We no longer need the Month and Hour columns so we can safely drop them.

# In[167]:


df = df.drop(['Month', 'Hour'], axis=1)


# In[168]:


df.head(5)


# ## Convert Value Field from mg to µg

# The US Embassy stores their PM2.5 values in mg/m^3. However when converting to the US EPA AQI standard the measurement used is µg/m^3. This is easy enough to convert by multiplying by 1,000. The purpose of this is to make it easy to display the results of the model and calculate AQI without any further computation.

# In[169]:


# 1 mg = 1,000 µg
df['Value'] = df.Value * 1000


# In[170]:


df.head(5)


# ## Convert TEMP and DEWP from F to C

# The source of the weather data comes from NOAA, which stores temperatures and dew points in Fahrenheit. As our end user is more familiar with Celsius we will convert these features now to eliminate the need to when displaying to the user.

# In[171]:


# Formula to convert F to C is: [°C] = ([°F] - 32) × 5/9
df['TEMP'] = (df.TEMP - 32) * 5.0/9.0


# In[172]:


# Formula to convert F to C is: [°C] = ([°F] - 32) × 5/9
df['DEWP'] = (df.DEWP - 32) * 5.0/9.0


# In[173]:


df.head(5)


# ## Convert SPD from Mph to Kph

# In[174]:


# 1 mph = 1.60934 kph
df['SPD'] = df.SPD * 1.60934


# In[175]:


df.head(5)


# ## Create day of the week feature

# In[176]:


import datetime as dt

df['day_week'] = df['Date'].dt.weekday_name


# In[177]:


df.head()


# In[178]:


df['day_week_cat'] = df.day_week.astype("category").cat.codes


# In[179]:


load = df[['Value', 'day_week']].groupby(['day_week']).mean()


# In[180]:


load.plot(kind='bar')


# In[181]:


df = df.drop(labels='day_week', axis=1)


# ## Previous hour values

# Given that pollution can take time to move through the city, and this movement is dependent on wind speed, direction, and total pollution level, it follows that the current air pollution level is somewhat determined by previous hours conditions. We can add these values to our data set relatively easily. 

# In[182]:


df.head(10)


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

# ### Get final dataset

# Before we move to the training and evaluating the model we can finalize our dataframe. The only unneeded feature we have remaining is the Date feature. 

# In[183]:


df.head(5)


# In[184]:


df = df.drop(['Date'], axis=1)
df.columns


# In[185]:


df.dtypes


# In[186]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# # Split into Training and Test Data

# Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

# ## Prepare Data Set for Regression

# The first models we will attempt will be regression models. We will prepare the our data frame by separating it into independent variables (X) and dependent variable being predicted (y).

# In[187]:


y = df['Value']
X = df.drop(['Value'], axis=1)


# In[188]:


X.head()


# In[189]:


y.head()


# In[190]:


#preprocess with scaler
from sklearn import preprocessing

#T = preprocessing.StandardScaler().fit_transform(X)
#T = preprocessing.MinMaxScaler().fit_transform(X)
#T = preprocessing.MaxAbsScaler().fit_transform(X)
#T = preprocessing.Normalizer().fit_transform(X)
T = X # No Change

X_s = pd.DataFrame(T, index=X.index, columns=X.columns)


# In[191]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=.3, random_state=0)


# In[192]:


X_train.shape, y_train.shape


# In[193]:


X_test.shape, y_test.shape


# ## Prepare Data Set for Classification

# Our predicted feature (y) is currently a continuous feature. This is what is required for the regression algorithms. However we also would like to apply classification algorithms to our data. This will require us to modify our Value (y) feature.

# In[194]:


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
# - The PM2.5 measurement and AQI are different numbers. The measurement is placed against a scale that is used to identify the AQI in the chart above.
# - AQI is a general scale that can combine multiple measures of air pollution. However in this case we will only be using PM2.5.

# The divider between the the AQI categories is called the breakpoint (or the end of the bin in other words). The breakpoint scale is the measurement of PM2.5 in µg/m^3. The official breakpoints between the categories has changed over time. We will be using the official breakpoints from the EPA. One thing to note is these breakpoints are used to derive the AQI value. We will be using them to simply assign the AQI category.

# | AQI Category | Low Breakpoint | High Breakpoint |
# | :--------------: | :----------------: | :-----------------: |
# | GOOD | 0 | 12 |
# | MODERATE | 12.1 | 35.4 |
# | UNHEALTHY FOR SENSITIVE | 35.5 | 55.4 |
# | UNHEALTHY |55.5 | 150.4 |
# | VERY UNHEALTHY | 150.5 | 250.4 |
# | HAZARDOUS | 250.5 | 350.4 |
# | HAZARDOUS | 350.5 | 500.4 |
# 
# Source: US EPA - https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html

# We will assign a numerical category to each AQI category from 1-5. Good being 1 and Hazardous being 6.

# In[195]:


# For our purposes hazardous is hazardous, no need to apply a different category.
y_cat = pd.cut(df['Value'],[-50,12,35.4,55.4,150.4,250.4,1000],labels=[1,2,3,4,5,6])


# Now we have a new dependent variable feature that is categorical. We can re-split our data using this new data set. 

# In[196]:


y_cat.dtypes


# In[197]:


from sklearn.model_selection import train_test_split

X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_s, y_cat, test_size=.3, random_state=0)


# In[198]:


X_train_cl.shape, y_train_cl.shape


# In[199]:


X_test_cl.shape, y_test_cl.shape


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

# In[200]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create linear regression object
regr = LinearRegression()


# In[201]:


# Train the model using the training sets
regr.fit(X_train, y_train)


# In[202]:


# Make predictions using the testing set
lin_pred = regr.predict(X_test)


# In[203]:


linear_regression_score = regr.score(X_test, y_test)
linear_regression_score


# In[204]:


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


# In[205]:


plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()


# ### Neural Network Regression

# In[206]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()


# In[207]:


# Train the model using the training sets
mlp.fit(X_train, y_train)


# In[208]:


# Score the model
neural_network_regression_score = mlp.score(X_test, y_test)
neural_network_regression_score


# In[209]:


# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)


# In[210]:


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


# In[211]:


X.columns


# In[212]:


plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()


# ### Decision Forest

# In[213]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=100)


# In[214]:


# Train the model using the training sets
regr_rf.fit(X_train, y_train)


# In[215]:


# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score


# In[216]:


# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)


# In[217]:


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, regr_rf_pred))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))


# In[260]:


df.columns


# In[218]:


plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()


# ### Decision Tree + AdaBoost

# In[219]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Decision Tree Regressor object
tree_1 = DecisionTreeRegressor()

tree_2 = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=200, learning_rate=.1)


# In[220]:


# Train the model using the training sets
tree_1.fit(X_train, y_train)
tree_2.fit(X_train, y_train)


# In[221]:


# Score the decision tree model
tree_1.score(X_test, y_test)


# In[222]:


# Score the boosted decision tree model
boosted_tree_score = tree_2.score(X_test, y_test)
boosted_tree_score


# In[223]:


# Make predictions using the testing set
tree_1_pred = tree_1.predict(X_test)
tree_2_pred = tree_2.predict(X_test)


# In[224]:


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, tree_2_pred))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, tree_2_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, tree_2_pred))


# In[225]:


plt.scatter(y_test, tree_1_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Tree Predicted vs Actual')
plt.show()


# In[226]:


plt.scatter(y_test, tree_2_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Boosted Decision Tree Predicted vs Actual')
plt.show()


# ## Classification

# ### Multiclass neural network

# In[227]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Create MLPClassifier object
mlp_cl = MLPClassifier(activation='logistic', random_state=1234)


# In[228]:


# Train the model using the training sets
mlp_cl.fit(X_train_cl, y_train_cl)


# In[229]:


# Score the model
multiclass_neural_network_score = mlp_cl.score(X_test_cl, y_test_cl)
multiclass_neural_network_score


# In[230]:


# Make predictions using the testing set
mnn_pred = mlp_cl.predict(X_test_cl)


# In[231]:


# Calculate probabilities
mnn_prob = mlp_cl.predict_proba(X_test_cl)


# In[232]:


# Calculate confusion matrix
confusion_mnn = confusion_matrix(y_test_cl,mnn_pred)


# In[233]:


confusion_mnn


# In[234]:


columns = ['Good','Moderate','Unhealthy for Sensitive','Unhealthy','Very Unhealthy','Hazardous']

plt.imshow(confusion_mnn, cmap=plt.cm.Blues, interpolation='nearest')
plt.xticks([0,1,2,3,4,5], columns, rotation='vertical')
plt.yticks([0,1,2,3,4,5], columns)
plt.colorbar()

plt.show()


# ### Multiclass logistic regression

# In[235]:


from sklearn.linear_model import LogisticRegression

# Create logistic regression object
log_regr = LogisticRegression()


# In[236]:


# Train the model using the training sets
log_regr.fit(X_train_cl, y_train_cl)


# In[237]:


# Score the model
logistic_regression_score = log_regr.score(X_test_cl, y_test_cl)
logistic_regression_score


# In[238]:


# Make predictions using the testing set
log_regr_pred = log_regr.predict(X_test_cl)


# In[239]:


# Calculate probabilities
log_regr_prob = log_regr.predict_proba(X_test_cl)


# In[240]:


# Calculate confusion matrix
confusion_log_regr = confusion_matrix(y_test_cl, log_regr_pred)


# In[241]:


confusion_log_regr


# In[242]:


columns = ['Good','Moderate','Unhealthy for Sensitive','Unhealthy','Very Unhealthy','Hazardous']

plt.imshow(confusion_log_regr, cmap=plt.cm.Blues, interpolation='nearest')
plt.xticks([0,1,2,3,4,5], columns, rotation='vertical')
plt.yticks([0,1,2,3,4,5], columns)
plt.colorbar()

plt.show()


# ### Random forest

# In[243]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forest Regressor object
cl_rf = RandomForestClassifier()


# In[244]:


# Train the model using the training sets
cl_rf.fit(X_train_cl, y_train_cl)


# In[262]:


# Score the model
cl_rf_score = cl_rf.score(X_test_cl, y_test_cl)
cl_rf_score


# In[246]:


# Make predictions using the testing set
cl_rf_pred = cl_rf.predict(X_test_cl)


# In[247]:


# Calculate confusion matrix
confusion_cl_rf = confusion_matrix(y_test_cl, cl_rf_pred)


# In[248]:


confusion_cl_rf


# In[249]:


columns = ['Good','Moderate','Unhealthy for Sensitive','Unhealthy','Very Unhealthy','Hazardous']

plt.imshow(confusion_cl_rf, cmap=plt.cm.Blues, interpolation='nearest')
plt.xticks([0,1,2,3,4,5], columns, rotation='vertical')
plt.yticks([0,1,2,3,4,5], columns)
plt.colorbar()

plt.show()


# ### Decision Tree

# In[250]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[251]:


# Create decision tree object with adaboost

tree_cl = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=300, learning_rate=.1)


# In[252]:


# Train the model using the training sets
tree_cl.fit(X_train_cl, y_train_cl)


# In[253]:


# Score the decision tree model
tree_cl_score = tree_cl.score(X_test_cl, y_test_cl)
tree_cl_score


# In[254]:


# Make predictions using the testing set
tree_cl_pred = tree_cl.predict(X_test_cl)


# In[255]:


# Calculate confusion matrix
confusion_tree_cl = confusion_matrix(y_test_cl, tree_cl_pred)


# In[256]:


confusion_tree_cl


# In[257]:


columns = ['Good','Moderate','Unhealthy for Sensitive','Unhealthy','Very Unhealthy','Hazardous']

plt.imshow(confusion_tree_cl, cmap=plt.cm.Blues, interpolation='nearest')
plt.xticks([0,1,2,3,4,5], columns, rotation='vertical')
plt.yticks([0,1,2,3,4,5], columns)
plt.colorbar()

plt.show()


# # Evaluate Models

# Let's compare all of our models scores. The regression and classification models are scored differently, with regression using r-squared and the classification models using an accuracy score. We can at the very least compare models of the same type.

# In[261]:


print("Regression models:")
print("Linear regression score: ", linear_regression_score)
print("Neural network regression score: ", neural_network_regression_score)
print("Decision forest score: ", decision_forest_score)
print("Boosted decision tree score: ", boosted_tree_score)
print("\n")
print("Classification models:")
print("Multiclass neural network score: ", multiclass_neural_network_score)
print("Logistic regression score: ", logistic_regression_score)
print("Random forest score: ", cl_rf_score)
print("Decision tree score: ", tree_cl_score)


# # Observations
# 
# - The best model is the multiclass neural network. 
# - Overall performance was quite poor for all of our models
# - We have not completed any hyperparameter tuning. This can be done in the next 
# - Our exploratory data analysis was mostly done through graphing and manual observation. Next steps would be to use more rigorous methods of exploration such as correlation, crosstabs, etc. 
# - Our direction feature is probably too granular. We have 17 classes of direction. However no connection between direction that are close to each other exist. 
# - We identified a few outlier points, however the range of the value feature is quite large and is a good candidate for further processing.
# - Our current data set ignores previous hours conditions and its impact on the current PM2.5 level.
# 
# Will address these observations in next draft.

# ## Analysis after run

# In[259]:


sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[268]:


create_feature_map(df)

