# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 09:03:01 2021

@author: aravi
"""

#%%#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from vam.whittaker import *
from sklearn.metrics.pairwise import haversine_distances
from sklearn.model_selection import train_test_split
import sklearn
import datetime
import random
import array
import math
import h5py







#%%
def getEnvelopeSegment(series, envparam = 0.0):
    
    #initialize
    success_flag = 0
    serieset = series.copy()
    
    #find_peaks
    p = find_peaks(series, height = 0.3,prominence = 0.03)[0]
    
    #envelope if there are multiple peaks
    if len(p)>1:
        success_flag=1
        diffs = [p[i]-p[0] for i in range(len(p))]
        dParam = max(diffs) +1 
    
        for i in range(len(p)-1):
            idx1 = p[i]
            idx2 = p[i+1]
            if idx2-idx1 < dParam and series[idx1] > envparam and series[idx2] > envparam:
                ser1 = series[idx1]
                ser2 = series[idx2]
                ifn = interp1d([idx1, idx2],[ser1,ser2])
                for j in range(idx1+1,idx2):
                    serieset[j] = ifn(j)
        
        seriese = [max(series[i],serieset[i]) for i in range(len(series))]
        seriese = pd.Series(seriese, index = series.index)     

    if success_flag == 0:
        ret_series = series
    else:
        ret_series = seriese
        
    return ret_series   

def getEnvelopeSeries(series, starts, ends):

    #initializaitons
    new_series = series.copy()

    #loop over segments
    for i in range(len(starts)):
        
        #extract segment
        segment = series[starts[i]:ends[i]]
        
        #create_envelope
        segment_e = getEnvelopeSegment(segment, envparam = 0.0)
        
        #replace values in series
        new_series[starts[i]:ends[i]] = segment_e
        
    return new_series    
        

def getAsymmetricWSSeries(series, lrange_min = -1, lrange_max = 2.5, param1 = 0.99):
    
    dates = series.index
    series = np.array(series)
    
    y = series.copy()

    # create weights
    w = np.array((y!=-3000)*1,dtype='double')
    
    #for i in range(w.shape[0]):
    #    if qa60[i] < 0.25:
    #            w[i] = 0.5
    
    lrange = array.array('d',np.linspace(lrange_min,lrange_max,11))
    
    # # apply whittaker filter with V-curve
    #zv, loptv = ws2doptv(y,w,lrange)
    
    #simple smoothing
    z = ws2d(y,100,w)
    
    # apply whittaker filter with V-curve and asymmetric smoothing
    zvp, loptvp = ws2doptvp(y,w,lrange,p=param1)
    
    #delete weights
    del w
    
    zvp =pd.Series(zvp, dates)
    z = pd.Series(z,dates)
    
    #return
    return zvp
    
def resample(series):
    
    dates = series.index
    
    sdate = dates[0]
    edate = dates[-1]
    
    upsampled = series.resample('D').interpolate(method = 'linear')
    
    return upsampled

def getNDVIPeaks(series, distance = 60, prominence = 0.1,height = 0.3):
    
    pdata = find_peaks(series, distance = distance, height = height, prominence = 0.01)
    peaks = pdata[0]
    peaks_p = pdata[1]['prominences']
    
    #intialize
    peaks_f = []
    
    
    for i in range(len(peaks)):
        if i == 0 or i == len(peaks)-1:
            peaks_f.append(peaks[i])
        else:
            if peaks_p[i] > prominence:
                peaks_f.append(peaks[i])
                
    return peaks_f

def refineTrough(series, peak, limit):

    """
    This function look for trough alternatives once some limits are identified

    """ 
    
    success_flag = 0
    levels = [0.2, 0.22, 0.25]
    
    if limit < peak:
        search_direction = 'left'
    elif limit >peak:
        search_direction = 'right'
    else:
        search_direction = 'center'
    
    for i in range(len(levels)):
        
        #pick search level
        level = levels[i]
        
        #find troughs of the level
        new_troughs = find_peaks(-1*series, height = -1*level)[0]

        #proceed if the length of troughs is greater than zero
        if len(new_troughs) > 0:
            
            if search_direction == 'left':
                candidates = [j for j in new_troughs if j < peak and j > limit]
            elif search_direction  == 'right':
                candidates = [j for j in new_troughs if j > peak and j < limit]
            else:
                candidates = [limit]
                
            if len(candidates) > 0:
                success_flag = 1
                break
    if success_flag == 1: 
        if search_direction  == 'left':
            ret_trough = candidates[-1]
        else:
            ret_trough = candidates[0]
            
    else:
        ret_trough =limit       
        
    return ret_trough    
    
    
def getMajorTroughs(series, peaks, start_idxs, end_idxs):
    
    #find all troughs
    troughs = find_peaks(-1*series, prominence = 0.03)[0]
    troughs = troughs.tolist()
    troughs.append(1)
    troughs.append(len(series)-2)
    troughs.sort()
    
        
    #intialize final_troughs
    troughs_left_final = []
    troughs_right_final = []
    
    #loop over all peaks
    for i in range(len(peaks)):
        #find all troughs before the peak
        if i == 0:
            prev_peak = 1
        else:
            prev_peak = peaks[i-1]
            
        #set search limits
        prev_limit = min(start_idxs[i], prev_peak)
            
        #find all the troughs on the left hand side    
        lhs_troughs = [j for j in troughs if j<= peaks[i] and j >= prev_limit]
        if len(lhs_troughs) ==0:
            proms_list = [0.02,0.01,0.0]
            for prom in proms_list: 
                new_troughs = find_peaks(-1*series, prominence = prom)[0]
                new_lhs_troughs = [k for k in new_troughs if k<= peaks[i] and k >= prev_limit]
                if len(new_lhs_troughs) > 0:
                    lhs_troughs  = new_lhs_troughs
                    break
                    
        
        #check if the start point is in the list and action based on this
        if 1 not in lhs_troughs:
            lhs_trough_vals = [series[j] for j in lhs_troughs]  
            min_trough = lhs_troughs[lhs_trough_vals.index(min(lhs_trough_vals))]
            min_trough = refineTrough(series, peaks[i], min_trough)
            troughs_left_final.append(min_trough)
        else:
            if len(lhs_troughs) > 1:
                lhs_troughs = lhs_troughs[1:]
                lhs_trough_vals = [series[j] for j in lhs_troughs if j !=1]  
                min_trough = lhs_troughs[lhs_trough_vals.index(min(lhs_trough_vals))]
                min_trough = refineTrough(series, peaks[i], min_trough)
                troughs_left_final.append(min_trough)
            else:
                troughs_left_final.append(1)
      
        
        if i != len(peaks)-1:
            next_limit = peaks[i+1]
        else:
            next_limit = min(end_idxs[i]+60, len(series)-2)
            
        #find rhs trough
        rhs_troughs = [j for j in troughs if j>=peaks[i] and j <= next_limit]
        
        if len(rhs_troughs) == 0 :
            rhs_troughs.append(next_limit)
    
        if len(series)-2 not in rhs_troughs:
            rhs_trough_vals = [series[j] for j in rhs_troughs]  
            min_trough = rhs_troughs[rhs_trough_vals.index(min(rhs_trough_vals))]
            min_trough = refineTrough(series, peaks[i], min_trough)
            troughs_right_final.append(min_trough)
            
        else:
            
            if len(rhs_troughs) > 1:
                rhs_troughs = rhs_troughs[:-1]
                rhs_trough_vals = [series[j] for j in rhs_troughs]  
                min_trough = rhs_troughs[rhs_trough_vals.index(min(rhs_trough_vals))]
                min_trough = refineTrough(series, peaks[i], min_trough)
                troughs_right_final.append(min_trough)
            else:
                troughs_right_final.append(rhs_troughs[0])
    

    return troughs_left_final, troughs_right_final  

def adjustSeasons(series, starts, peaks, ends):
    
    #initializations
    starts_final = []
    ends_final = []
    durations_final = []
    
    #adjust starts
    for i in range(len(starts)):
        if (series[peaks[i]] - series[starts[i]]) > 0.1 and (ends[i] - starts[i]) > 120:
            critical_val = series[starts[i]] + (series[peaks[i]] - series[starts[i]])*0.2
            adj_start = [j for j in range(starts[i], peaks[i]) if series[j]< critical_val and series[j+1]>= critical_val]
            starts_final.append(adj_start[0])
        else:
            starts_final.append(starts[i])
            
    #adjust ends
    for i in range(len(ends)):
        hvst_param = 0.2
        if (series[peaks[i]] - series[ends[i]]) > 0.1 and (ends[i] - starts[i]) > 120:
            critical_val = max(series[peaks[i]] - (series[peaks[i]] - series[ends[i]])*0.8, hvst_param)
            adj_end = [j for j in range( peaks[i], ends[i]) if series[j]> critical_val and series[j+1]<= critical_val]
            ends_final.append(adj_end[-1])
        else:
            ends_final.append(ends[i])
    
    #get season durations    
    for i in range(len(starts_final)):
        durations_final.append(ends_final[i]-starts_final[i])
                 
    return starts_final, ends_final, durations_final    
        
       
def getSeasons(vhvv, ndvi):
    
    #copy
    series_vhvv = vhvv.copy()
    series_ndvi = ndvi.copy()
    
    #smooth the vhvv plot using very heavy smoothing
    series_vhvv_s= getAsymmetricWSSeries(series_vhvv,lrange_min = -1, lrange_max = 2.5, param1 = 0.95)
    series_ndvi= getAsymmetricWSSeries(series_ndvi,lrange_min = -1, lrange_max = 2.5, param1 = 0.95)
    
    #peaks and troughs
    vhvv_peaks = find_peaks(series_vhvv_s, distance = 75, prominence = 0.03)[0]
    
    vhvv_troughs = find_peaks(-1*series_vhvv_s, distance = 75, prominence = 0.03)[0]
    
    #update intial estimates of seasons ends as the vhvv peaks
    season_starts = vhvv_troughs.tolist()   
    
    if vhvv_peaks[0]<vhvv_troughs[0]:
        season_starts.append(1)   
    
    #add in the last point
    season_starts.append(len(series_ndvi)-2)
    
    season_starts.sort()
      
    #find season peaks by finding the closest NDVI peaks
    season_peaks_final = []
    season_starts_final = []
    season_ends_final= []
    
    #get the NDVI peaks that need to be mapped to the season ends
    ndvi_peaks = getNDVIPeaks(series_ndvi, distance = 60, height = 0.3, prominence = 0.1)
    
    #add in the final point
    if series_ndvi[-1] > 0.4:
        ndvi_peaks.append(len(series_ndvi)-2)
    
    #initialize search_ranges
    start_idxs = []
    end_idxs = []
    
    #cycle over the season ends and adjust
    for i in range(len(season_starts)-1):
        start_idx = season_starts[i]
        end_idx = season_starts[i+1]
        segment_ndvi_peaks = [j for j in ndvi_peaks if j > start_idx and j<= end_idx]
        if len(segment_ndvi_peaks) == 1:
            season_peaks_final.append(segment_ndvi_peaks[0])
            start_idxs.append(start_idx)
            end_idxs.append(end_idx)
        elif len(segment_ndvi_peaks) > 1 :
            segment_ndvi_peaks_vals = [series_ndvi[j] for j in segment_ndvi_peaks]
            max_val_index = segment_ndvi_peaks_vals.index(max(segment_ndvi_peaks_vals))
            season_peaks_final.append(segment_ndvi_peaks[max_val_index])
            start_idxs.append(start_idx)
            end_idxs.append(end_idx)
        
    #get the NDVI troughs between the peaks
    ndvi_left_troughs, ndvi_right_troughs = getMajorTroughs(series_ndvi, season_peaks_final, start_idxs, end_idxs)
            
    # season starts and ends
    season_starts_final = ndvi_left_troughs
    season_ends_final = ndvi_right_troughs   
        
    #get segmentwise envelopes
    ndvi_env = getEnvelopeSeries(series_ndvi, season_starts_final, season_ends_final)
        
    #figure out adjusted season starts
    season_starts, season_ends, season_durations = adjustSeasons(series_ndvi, season_starts_final,season_peaks_final, season_ends_final)
    
    keep_idxs = []
    #delete short seasons
    for i in range(len(season_durations)):
        if season_durations[i] > 60:
            keep_idxs.append(i)
            
    #final trim
    season_starts = [season_starts[i] for i in keep_idxs] 
    season_ends = [season_ends[i] for i in keep_idxs] 
    season_durations = [season_durations[i] for i in keep_idxs]        
    season_peaks = [season_peaks_final[i] for i in keep_idxs]    
    
    return season_starts, season_ends, season_peaks, season_durations


def CropClassIndex(msidata, sardata):
    # compute msi indices
    msidata['date'] = pd.to_datetime(msidata['date'],infer_datetime_format =True)
    msidata.sort_values(by = 'date', ascending = True, inplace = True)
    msidata.set_index(keys = 'date',drop = True, inplace = True)
    msidata = msidata.groupby('date').mean()
    msidata['ndvi']= (msidata['B8'] - msidata['B4'])/(msidata['B8'] + msidata['B4']) #save
    msidata['ndmi'] = (msidata['B8A'] - msidata['B11'])/(msidata['B8A'] + msidata['B11'])  #save
    msidata['gndvi'] = (msidata['B8'] - msidata['B3'])/(msidata['B8'] + msidata['B3'])  #save
    msidata['nmdi'] = (msidata['B8'] - (msidata['B11']-msidata['B12']))/(msidata['B8'] + (msidata['B11']-msidata['B12']))  #save
    msidata.dropna(axis = 0, how = 'any', inplace=True)
    msidata['QA60'] = msidata['QA60']/2048.0
    msidata = msidata[msidata['QA60'] < 0.25]
   
    # sar data
    sardata['date'] = pd.to_datetime(sardata['date'],infer_datetime_format =True)
    sardata.sort_values(by = 'date', ascending = True, inplace = True)
    sardata.set_index(keys = 'date',drop = True, inplace = True)
    sardata = sardata.groupby(level = 'date').mean()
    sardata.dropna(axis = 0, how = 'any', inplace=True)
   
    # find days difference in msidata
    msidatediff = [(i-j).days for i, j in zip(msidata.index[1:], msidata.index[:-1])]
    sardatediff = [(i-j).days for i, j in zip(sardata.index[1:], sardata.index[:-1])]
   
   
    #get and harmonize data
    ndvi = msidata['ndvi']
    ndvi = resample(ndvi)
   
    vhvv = sardata['VHbyVV']
    vhvv = resample(vhvv)
   
    gndvi = msidata['gndvi']
    gndvi = resample(gndvi)
   
    # ndmi = msidata['ndmi']
    # nmdi = msidata['nmdi']
   
    vh = sardata['VH']
    vh = resample(vh)
   
    vv = sardata['VV']
    vv = resample(vv)
   
    # https://earth.esa.int/eogateway/documents/20142/0/ers-sar-calibration-issue2-5f.pdf
    rvi = 4*(10**(vh/10))/(10**(vh/10) + 10**(vv/10)) # 10^log10(DN) = DN; DN = 10^(dB/10)
    rvi = resample(rvi)
   
    # # Smooth nmdi and ndmi index
    # dates = nmdi.index
    # nmdi = pd.Series(nmdi,dates)
    # nmdi = getAsymmetricWSSeries(nmdi, lrange_min = -1, lrange_max = 0, param1 = 0.01)
    # nmdi = resample(nmdi)
   
    # dates = ndmi.index
    # ndmi = savgol_filter(ndmi, window_length = 5, polyorder = 1)
    # ndmi = pd.Series(ndmi,dates )
    # ndmi = getAsymmetricWSSeries(ndmi, lrange_min = -1, lrange_max = 0, param1 = 0.5)
    # ndmi = resample(ndmi)
   
    # Common date
    sDate = max(vhvv.index[0], ndvi.index[0])
    eDate = min(vhvv.index[-1], ndvi.index[-1])
    sDate_str = sDate.strftime('%Y-%m-%d')
    eDate_str = eDate.strftime('%Y-%m-%d')
    ndvi = ndvi[sDate_str:eDate_str]
    vhvv = vhvv[sDate_str:eDate_str]
    gndvi = gndvi[sDate_str:eDate_str]
    # nmdi = nmdi[sDate_str:eDate_str]
    # ndmi = ndmi[sDate_str:eDate_str]
    vh = vh[sDate_str:eDate_str]
    vv = vv[sDate_str:eDate_str]
    rvi = rvi[sDate_str:eDate_str]
   
    return ndvi, vhvv, gndvi, msidatediff, sardatediff

#%%

def findDate(s_starts, s_ends, s_date, h_date, ndvi):
    
    temp1=s_starts.copy()
    temp2=s_ends.copy()
    
    temp1days=[]
    temp2days=[]
    
    ##Converts to timestamp
    sowing_date_time=pd.to_datetime(s_date)
    harvest_date_time=pd.to_datetime(h_date)
        
    #Convert season start and end from list to timestamp
    for i in range(len(temp1)):
        temp1[i]=ndvi.index[temp1[i]]
        temp2[i]=ndvi.index[temp2[i]]
        
    for i in range(len(temp1)):
        temp1days.append(abs((temp1[i]-sowing_date_time).days))
        temp2days.append(abs((temp2[i]-harvest_date_time).days))
        
    lst1 = np.asarray(temp1days)
    #lst2 = np.asarray(temp2days)
    idx1 = lst1.argmin()
    #idx2 = lst2.argmin()
    
    #idx=max(idx1,idx2)
    return ndvi.index[s_starts[idx1]], ndvi.index[s_ends[idx1]]

    
    


#%%

def resampledataframe(df_temp):
    
    df_return=pd.DataFrame()

    for i in df_temp.columns:
        dates = df_temp[i].index
    
        sdate = dates[0]
        edate = dates[-1]
        
        upsampled = df_temp[i].resample('D').interpolate(method = 'linear')
        
        df_return[i]=upsampled
    return df_return

        
#%%

def sections(df_ndvi,df_msi,df_weather,df_vhvv,duration,x):
    
    values={}
    length=math.ceil(duration/x)
    c=0
    for i in range(0,duration,length):
        end=i+length 
        if end>duration:
            end=duration
        
        values['NDVI_Max'+str(c)]=df_ndvi[i:end].max()
        values['NDVI_Median'+str(c)]=df_ndvi[i:end].median()
        
        values['B7/B6_Median'+str(c)]=(df_msi[i:end]['B7']/df_msi[i:end]['B6']).median()
        
        values['VHVV_Max'+str(c)]=df_vhvv[i:end].max()
        values['VHVV_Median'+str(c)]=df_vhvv[i:end].median()
        
        values['RainFall_Median'+str(c)]=df_weather[i:end]['RainFall'].median()
        values['TempMax_Median'+str(c)]=df_weather[i:end]['TempMAX'].median()
        values['TempMin_Median'+str(c)]=df_weather[i:end]['TempMIN'].median()
        
    
        c=c+1
        
    
    return values
#%%


def get_seasons_dates(sowing, ndvi, season_starts, season_ends):
   
    year = sowing.year
    month = sowing.month
    if month == 1:
        year = year-1
   
    season = None
    date_range = None
   
    for ss, se in zip(season_starts, season_ends):
       
        #sowing = ndvi.index[0] + pd.Timedelta(days=ss)
        k_start1 = pd.to_datetime(str(year)+'-05-01')
        k_start2 = pd.to_datetime(str(year)+'-08-31')
        r_start1 = pd.to_datetime(str(year)+'-09-01')
        r_start2 = pd.to_datetime(str(year)+'-12-31')
        r_start3 = pd.to_datetime(str(year+1)+'-01-01')
        r_start4 = pd.to_datetime(str(year+1)+'-01-31')
        z_start1 = pd.to_datetime(str(year)+'-02-01')
        z_start2 = pd.to_datetime(str(year)+'-04-30')
       
        # check estimated season vs actual crop season
        if (ndvi.index[0] + pd.Timedelta(days=ss) >= k_start1) and (ndvi.index[0] + pd.Timedelta(days=ss) <= k_start2):
            if 5 <= month <= 8:
                season = 'Kharif'
                date_range = [ndvi.index[0], ndvi.index[-1]]
        elif ((ndvi.index[0] + pd.Timedelta(days=ss) >= r_start1) and (ndvi.index[0] + pd.Timedelta(days=ss) <= r_start2)) or ((ndvi.index[0] + pd.Timedelta(days=ss) >= r_start3) and (ndvi.index[0] + pd.Timedelta(days=ss) <= r_start4)):
            if (9 <= month <= 12) or (month == 1):
                season = 'Rabi'
                date_range = [ndvi.index[0], ndvi.index[-1]]
        elif (ndvi.index[0] + pd.Timedelta(days=ss) >= z_start1) and (ndvi.index[0] + pd.Timedelta(days=ss) <= z_start2):
            if 2 <= month <= 4:
                season = 'Zaid'
                date_range = [ndvi.index[0], ndvi.index[-1]]
        else:
            #print ('new season!')
            pass
           
    return season, date_range

#%%
data=r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset'

RSdata=r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\RSdataNew'

Weatherdata=r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\WeatherDataNew'

Metadata=r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\Crop_YieldData20210826_FINAL.csv'

FilePartial=r'C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset\FilePartial.csv'
df=pd.read_csv(Metadata)
count=0
print(df.head(5))
total_rows=len(df.index)
print(total_rows)

v=[]
k=[]
x=[]
y=[]
z=[]
difference=[]


df_final=pd.DataFrame()
df_yield=pd.DataFrame(columns=['Yield','Bengal Gram(Gram)(Whole)','Bajra(Pearl Millet/Cumbu)','Cotton(Kapas)','Green Gram (Moong)(Whole)','Maize','Millets','Onion','Paddy(Dhan)(Common)','Soyabean','Wheat'])

#%%

 def crop_metadata(wktcropsowdate):
    # crop dataframe
    crop_df = pd.read_csv(wktcropsowdate)
    try:
        crop_df['sowing_date']=pd.to_datetime(crop_df['sowing_date'], format="%d-%m-%Y")
        crop_df['harvest_date']=pd.to_datetime(crop_df['harvest_date'], format="%d-%m-%Y")
    except:
        crop_df['sowing_date']=pd.to_datetime(crop_df['sowing_date'], format="%Y-%m-%d")
        crop_df['harvest_date']=pd.to_datetime(crop_df['harvest_date'], format="%Y-%m-%d")
       
    crop_df['plot_id'] = crop_df['plot_id'].astype(str)
    return crop_df

#%%
    
field_data = crop_metadata(Metadata)


no_files = []
partial_data = []
no_seasons = []

#%%


def FileSize(filename):
    res = os.stat(filename)
    fsize_bytes = res.st_size
    fsize_kb = (fsize_bytes >> 10)
    return fsize_kb

#%%



for ix, row in field_data.iterrows():
    fpo_id = row['fpo_id']
    plot_id = row['plot_id']
    crop_name = row['crop_name']
    crop_season = row['season']
    sowing_date = row['sowing_date']
    harvest_date = row['harvest_date']
    crop_duration = row['duration']
    #state = row['state']
   
    #print (plot_id, crop_season, sowing_date, harvest_date)
   
    msifile = 'C:\\Users\\aravi\\Documents\\Dvara-Internship\\Crop Yield Estimation\\Dataset\\RSdataNew\\FPO_'+str(fpo_id)+'\\'+str(plot_id)+'_MSI_data_group.csv'
    sarfile = 'C:\\Users\\aravi\\Documents\\Dvara-Internship\\Crop Yield Estimation\\Dataset\\RSdataNew\\FPO_'+str(fpo_id)+'\\'+str(plot_id)+'_SAR_data_group.csv'
   
    isfile = os.path.isfile(msifile)
   
    if isfile == True:
        msi_fsize = FileSize(msifile)
        sar_fsize = FileSize(sarfile)
       
        if msi_fsize > 35 and sar_fsize > 7:
           
            msidata = pd.read_csv(msifile)
            sardata = pd.read_csv(sarfile)
           
            # get resample raw indexes
            ndvi, vhvv, gndvi, msidatediff, sardatediff = CropClassIndex(msidata, sardata)
           
            ndvi_raw_res = ndvi.copy()
            vhvv_raw_res = vhvv.copy()
            gndvi_raw_res = gndvi.copy()
           
            # smooth ndvi and vhvv to identify seasons
            dates = vhvv.index
            vhvv = savgol_filter(vhvv, window_length = 51, polyorder = 1)
            vhvv = pd.Series(vhvv, dates)
           
            dates = ndvi.index
            ndvi = savgol_filter(ndvi, window_length = 51, polyorder = 1)
            ndvi = pd.Series(ndvi, dates)
           
            # whittaker
            ndvi_smooth = getAsymmetricWSSeries(ndvi, lrange_min = -1, lrange_max = 2.5, param1 = 0.99)
           
            # get seasons
            season_starts, season_ends, season_peaks, season_durations = getSeasons(vhvv, ndvi)
                  
            # get crop season date range
            season_est, date_ranges = get_seasons_dates(sowing_date, ndvi_smooth, season_starts, season_ends)
           
            if season_est != None:
                pass
            else:
                #print (fpo_id, plot_id, crop_season, sowing_date, harvest_date)
                no_seasons.append([fpo_id, plot_id])
        else:
            partial_data.append([fpo_id, plot_id])
    else:
        no_files.append([fpo_id, plot_id])



