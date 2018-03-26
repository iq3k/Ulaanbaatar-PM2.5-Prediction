# Predicting PM2.5 Pollution in Ulaanbaatar

## Introduction
This project aims to predict PM2.5 levels in Ulaanbaatar, the capital city of Mongolia. Ulaanbaatar is the coldest capital city on the planet and also has some of the worst pollution. It's location in a valley and lack of infrastructure mean that the majority of the population use raw coal for heat and cooking during the long and severe winter. 

Several machine learning models were tested. A random forest regression model was selected. A full write up of the project can be found on Medium in four parts:
- [Part 1, Introduction to the problem and some sollutions](https://medium.com/roberts-data-stories/ulaanbaatar-air-pollution-part-1-35e17c83f70b)
- Part 2, Exploring the data
- Part 3, The machine learning model
- Part 4, Deployment


## Reproducibility
The Jupyter notebook contains the exploratory data analysis, data cleaning, and algorithm testing. Notebook is written in Python 3.6. Requirements.txt lists all dependencies to run the code. The original data is included (**weather-and-aqi-v5.csv**). This is a combination of weather data and pollution data. 

Weather data was obtained from NOAA at their global hourly data access tool [link](https://www.ncei.noaa.gov/access-ui/data-search?datasetId=global-hourly). Pollution data is from the US Embassy in Ulaanbaatar's PM2.5 monitoring station [link](https://www.stateair.mn/).

## Production Model
The production machine learning model has been deployed on Microsofts's Azure ML platform. The model has been published to the Studio Gallery and can be found [here](https://gallery.cortanaintelligence.com/Experiment/UB-PM2-5-Regression-2).
