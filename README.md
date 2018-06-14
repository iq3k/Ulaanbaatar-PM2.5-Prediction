# Predicting PM2.5 Pollution in Ulaanbaatar

_Live model now at:_ [PM25.mn](https://pm25.mn)

## Introduction
This project aims to predict PM2.5 levels in Ulaanbaatar, the capital city of Mongolia. Ulaanbaatar is the coldest capital city on the planet and also has some of the worst pollution. It's location in a valley and lack of infrastructure mean that the majority of the population use raw coal for heat and cooking during the long and severe winter. 

Several machine learning models were tested. A random forest regression model was selected. A full write up of the project can be found on Medium in four parts:
- [Part 1, Introduction to the problem and some solutions](https://medium.com/roberts-data-stories/ulaanbaatar-air-pollution-part-1-35e17c83f70b)
- [Part 2, Exploring the data](https://medium.com/mongolian-data-stories/air-pollution-part-2-f9f4da33a1bd)
- [Part 3, The machine learning model](https://medium.com/mongolian-data-stories/part-3-the-model-b2fb9a25a07c)
- [Part 4, Deployment](https://medium.com/mongolian-data-stories/predicting-pm2-5-using-machine-learning-part-4-deployment-54086b5354d1)

The goal of this project is to provide citizens of Ulaanbaatar a tool to use in protecting themselves and their families from air pollution. In testing our RMSE was 28 (scale is 0-500). This is sufficient to enable the prediction of the AQI category.


## Reproducibility
The Jupyter notebook contains the exploratory data analysis, data cleaning, and algorithm testing. Notebook is written in Python 3.6. Requirements.txt lists all dependencies to run the code. The original data is included (**weather-and-aqi-v5.csv**). This is a combination of weather data and pollution data. Date range for data: 10-1-2015 to  1-31-2018

Weather data was obtained from NOAA at their global hourly data access tool [link](https://www.ncei.noaa.gov/access-ui/data-search?datasetId=global-hourly). Pollution data is from the US Embassy in Ulaanbaatar's PM2.5 monitoring station [link](https://www.stateair.mn/).

## Production Model
The production machine learning model has been deployed on Microsofts's Azure ML platform. The model has been published to the Studio Gallery and can be found [here](https://gallery.cortanaintelligence.com/Experiment/UB-PM2-5-Regression-2).

Live predictions can be found at: [PM25.mn](https://pm25.mn)


Big thanks to Amarbayan for writing the backend scripts, database management architecture, and assisting with making the front end site.
