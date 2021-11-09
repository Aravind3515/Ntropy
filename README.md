# CROP YIELD ESTIMATION

## Project Goal
Estimate the crop yield for specific crops on farmland. Based on this information farmers can decide what crops to prioritize. This product would help banks who can decide how much in loans to give to farmers based on thee yield of the farmers plot.


## Data
The dataset includes spectral bands which are downloaded from the Sentinel-2 satellite through an open-source platform. 

Data for a 1000 plots were collected to train the model. For previous years data the yield produced by each plot for a crop over a season was recorded Data collected included the FPO ID, Plot ID, Latitude, Longitude etc. With the latitude and longitude of the plot we could then download the remote sensing data for that plot


For each plot that has a yield(provided by the farmers) the remote sensing data would be downloaded in the form of time series.Each plot of land has 3 years of MSI, SAR and Weather data recorded


## Files
There was errors while downloading the data so we had to find the plots with the errors and redownload the data- Missing Files.py

To decide on what features to use for our model we used XgBoost- Feature Extraction.py

We prepared the initial dataset- OriginalDataSetPrepCode.py but we got poor results due to lack of data

To improve the accuracy of the model we applied data augmentation techniques and trained the final model- FinalDataSet-Model.py

## Results

Testing MAE: 0.092

Correlation: 0.891


