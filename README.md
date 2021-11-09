**Project Goal**-Estimate the crop yield for specific crops on farmland. Based on this information farmers can decide how much to ask for loans from banks or decide what crops to prioritize. This product would help farmers as well as this product could also be given to banks who can decide to give loans to farmers based on these yield of the farmers plot.



The dataset includes spectral bands which are downloaded from the Sentinel-2 satellite through an open-source platform. 

Data for a 1000 plots were collected to train the model. For previous years data the yield produced by each plot for a crop over a season was recorded Data collected included the FPO ID, Plot ID, Latitude, Longitude etc. With the latitude and longitude of the plot we could then download the remote sensing data for that plot


For each plot that has a yield(provided by the farmers) the remote sensing data would be downloaded in the form of time series.Each plot of land has 3 years of MSI, SAR and Weather data recorded


There was errors whild downloading the data so we had to find the plots with the errors and redownload the data- Missing Files.py

To decide on what features to use for our model we used XgBoost- Feature Extraction.py

We prepared the initial dataset- OriginalDataSetPrepCode.py but we got poor results

To improve the accuracy of the model we applied data augmentation techniques and trained the final model- FinalDataSet-Model.py


