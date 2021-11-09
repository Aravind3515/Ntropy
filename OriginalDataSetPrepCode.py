# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 07:48:19 2021

@author: aravi
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from vam.whittaker import *
import array
import math
import h5py
import datetime

os.chdir(r"C:\Users\aravi\Documents\Dvara-Internship\Crop Yield Estimation\Dataset")

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

#%%

def FileSize(filename):
    res = os.stat(filename)
    fsize_bytes = res.st_size
    fsize_kb = (fsize_bytes >> 10)
    return fsize_kb

def CropClassIndex(msidata, sardata, whtdata):
    # compute msi indices
    msidata['date'] = pd.to_datetime(msidata['date'],infer_datetime_format =True)
    msidata.sort_values(by = 'date', ascending = True, inplace = True)
    msidata.set_index(keys = 'date',drop = True, inplace = True)
    msidata = msidata.groupby('date').mean()
    msidata['ndvi']= (msidata['B8'] - msidata['B4'])/(msidata['B8'] + msidata['B4']) #save
    msidata['ndmi'] = (msidata['B8A'] - msidata['B11'])/(msidata['B8A'] + msidata['B11'])  #save
    msidata['gndvi'] = (msidata['B8'] - msidata['B3'])/(msidata['B8'] + msidata['B3'])  #save
    msidata['nmdi'] = (msidata['B8'] - (msidata['B11']-msidata['B12']))/(msidata['B8'] + (msidata['B11']-msidata['B12']))
    msidata['evi'] = 2.5 * (msidata['B8'] - msidata['B4']) / ((msidata['B8'] + 6.0 * msidata['B4'] - 7.5 * msidata['B2']) + 1.0)
    msidata['savi'] = (msidata['B8'] - msidata['B4']) / (msidata['B8'] + msidata['B4'] + 0.5) * (1.0 + 0.5)
    msidata['B7byB6'] = msidata['B7']/msidata['B6']
    msidata.dropna(axis = 0, how = 'any', inplace=True)
    msidata['QA60'] = msidata['QA60']/2048.0
    msidata = msidata[msidata['QA60'] < 0.25]
    
    # sar data
    sardata['date'] = pd.to_datetime(sardata['date'],infer_datetime_format =True)
    sardata.sort_values(by = 'date', ascending = True, inplace = True)
    sardata.set_index(keys = 'date',drop = True, inplace = True)
    sardata = sardata.groupby(level = 'date').mean()
    sardata.dropna(axis = 0, how = 'any', inplace=True)
    
    # weather data
    whtdata['Date'] = pd.to_datetime(whtdata['Date'], infer_datetime_format =True)
    whtdata.sort_values(by = 'Date', ascending = True, inplace = True)
    whtdata.set_index(keys = 'Date',drop = True, inplace = True)
    #whtdata.dropna(axis = 0, how = 'any', inplace=True)
    
    
    # find days difference in msidata
    ##msidatediff = [(i-j).days for i, j in zip(msidata.index[1:], msidata.index[:-1])]
    ##sardatediff = [(i-j).days for i, j in zip(sardata.index[1:], sardata.index[:-1])]
    
    
    #get and harmonize data
    ndvi = msidata['ndvi']
    ndvi = resample(ndvi)
    
    vhvv = sardata['VHbyVV']
    vhvv = resample(vhvv)
    
    gndvi = msidata['gndvi']
    gndvi = resample(gndvi)
    
    b7b6 = msidata['B7byB6']
    b7b6 = resample(b7b6)
    
    rainfall = whtdata['RainFall']
    #rainfall = resample(rainfall)
    tempmax = whtdata['TempMAX']
    #tempmax = resample(tempmax)
    tempmin = whtdata['TempMIN']
    #tempmin = resample(tempmin)
            
    # Common date
    sDate = max(vhvv.index[0], ndvi.index[0])
    eDate = min(vhvv.index[-1], ndvi.index[-1])
    sDate_str = sDate.strftime('%Y-%m-%d')
    eDate_str = eDate.strftime('%Y-%m-%d')
    ndvi = ndvi[sDate_str:eDate_str]
    vhvv = vhvv[sDate_str:eDate_str]
    gndvi = gndvi[sDate_str:eDate_str]
    b7b6 = b7b6[sDate_str:eDate_str]
    rainfall = rainfall[sDate_str:eDate_str]
    tempmax = tempmax[sDate_str:eDate_str]
    tempmin = tempmin[sDate_str:eDate_str]
        
    return ndvi, vhvv, b7b6, rainfall, tempmax, tempmin


def crop_metadata(wktcropsowdate):
    # crop dataframe
    crop_df = pd.read_excel(wktcropsowdate)
    try:
        crop_df['sowing_date']=pd.to_datetime(crop_df['sowing_date'], format="%d-%m-%Y")
        crop_df['harvest_date']=pd.to_datetime(crop_df['harvest_date'], format="%d-%m-%Y")
    except:
        crop_df['sowing_date']=pd.to_datetime(crop_df['sowing_date'], format="%Y-%m-%d")
        crop_df['harvest_date']=pd.to_datetime(crop_df['harvest_date'], format="%Y-%m-%d")
        
    crop_df['plot_id'] = crop_df['plot_id'].astype(str)
    return crop_df  

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
                date_range = [ndvi[ss:se].index[0], ndvi[ss:se].index[-1]]
        elif ((ndvi.index[0] + pd.Timedelta(days=ss) >= r_start1) and (ndvi.index[0] + pd.Timedelta(days=ss) <= r_start2)) or ((ndvi.index[0] + pd.Timedelta(days=ss) >= r_start3) and (ndvi.index[0] + pd.Timedelta(days=ss) <= r_start4)):
            if (9 <= month <= 12) or (month == 1):
                season = 'Rabi'
                date_range = [ndvi[ss:se].index[0], ndvi[ss:se].index[-1]]
        elif (ndvi.index[0] + pd.Timedelta(days=ss) >= z_start1) and (ndvi.index[0] + pd.Timedelta(days=ss) <= z_start2):
            if 2 <= month <= 4:
                season = 'Zaid'
                date_range = [ndvi[ss:se].index[0], ndvi[ss:se].index[-1]]
        else:
            #print ('new season!')
            pass
            
    return season, date_range

def sections(ndvi, vhvv, b7b6, rnfl, tmax, tmin): 
    values = {}
    c=0
    for i in range(len(ndvi)):
        
        values['NDVI_Max'+str(c)] = ndvi[i].max()
        values['NDVI_Median'+str(c)] = ndvi[i].median()
    
        values['VHVV_Max'+str(c)] = vhvv[i].max()
        values['VHVV_Median'+str(c)] = vhvv[i].median()
    
        values['B7B6_Median'+str(c)] = b7b6[i].median()
    
        values['RainFall_Median'+str(c)] = rnfl[i].median()
        values['TempMax_Median'+str(c)] = tmax[i].median()
        values['TempMin_Median'+str(c)] = tmin[i].median()
        
        c=c+1
        
    return values
#%%

def chunks(mylist, num):
    n = int(len(mylist)/num)
    m = len(mylist) - n*num
    pair = [[i*n, i*n+n] for i in range(num)]
    if pair[-1][1] < len(mylist):
        pair[-1][1] = pair[-1][1] + m
    subset = [mylist[p[0]:p[1]] for p in pair]
    return subset
#%%
cropfinalplots = "FinalDataset-2.xlsx"

# final/selected crop plots
pdf = pd.read_excel(cropfinalplots)

crop_list = pdf['crop_name'].unique().tolist()

season_list = pdf['season'].unique().tolist()

field_data = crop_metadata(cropfinalplots)

finaldata=pd.DataFrame()

#%%

no_files = []
partial_data = []
no_seasons = []
dflist = []
season_length = []
duration_length = []
res=[]
count=0
missing=0
X=np.zeros((651,215,17))
y=np.zeros((651,1))

for ix, row in field_data.iterrows():
    fpo_id = row['fpo_id']
    plot_id = row['plot_id']
    crop_name = row['crop_name']
    crop_season = row['season']
    sowing_date = row['sowing_date']
    harvest_date = row['harvest_date']
    #crop_duration = row['duration']
    crop_yield = row['yield_per_acre']
    
    wkt=row['wkt']
    
    #print (plot_id, crop_season, sowing_date, harvest_date)
    
    msifile = 'RSdataNew\\FPO_'+str(fpo_id)+'\\'+str(plot_id)+'_MSI_data_group.csv'
    sarfile = 'RSdataNew\\FPO_'+str(fpo_id)+'\\'+str(plot_id)+'_SAR_data_group.csv'
    weatherfile = 'WeatherDataNew\\Wth_'+str(fpo_id)+'_'+str(plot_id)+'.csv'
    
    isfilemsi = os.path.isfile(msifile)
    isfilesar = os.path.isfile(sarfile)
    isfileweather = os.path.isfile(weatherfile)
    
    
    if isfilemsi == True and isfilesar == True:
        msi_fsize = FileSize(msifile)
        sar_fsize = FileSize(sarfile)
        
        if msi_fsize > 35 and sar_fsize > 7:
            
            msidata = pd.read_csv(msifile)
            sardata = pd.read_csv(sarfile)
            whtdata = pd.read_csv(weatherfile)
            
            # get resample raw indexes
            ndvi, vhvv, b7b6, rainfall, tempmax, tempmin = CropClassIndex(msidata, sardata, whtdata)
            
            ndvi_raw_res = ndvi.copy()
            vhvv_raw_res = vhvv.copy()
            b7b6_raw_res = b7b6.copy()
            
            # smooth ndvi and vhvv to identify seasons
            dates = vhvv.index
            vhvv = savgol_filter(vhvv, window_length = 47, polyorder = 1)
            vhvv = pd.Series(vhvv, dates)
            
            dates = ndvi.index
            ndvi = savgol_filter(ndvi, window_length = 51, polyorder = 1)
            ndvi = pd.Series(ndvi, dates)
            
            dates = b7b6.index
            b7b6 = savgol_filter(b7b6, window_length = 51, polyorder = 1)
            b7b6 = pd.Series(b7b6, dates)
            
            # get seasons
            season_starts, season_ends, season_peaks, season_durations = getSeasons(vhvv, ndvi)
            
            # whittaker
            ndvi_smooth = getAsymmetricWSSeries(ndvi, lrange_min = -1, lrange_max = 2.5, param1 = 0.99)
            b7b6_smooth = getAsymmetricWSSeries(b7b6, lrange_min = -1, lrange_max = 2.5, param1 = 0.99)
            vhvv_smooth = getAsymmetricWSSeries(vhvv, lrange_min = -1, lrange_max = 2.5, param1 = 0.5)
                   
            # get crop season date range
            season_est, date_ranges = get_seasons_dates(sowing_date, ndvi_smooth, season_starts, season_ends)
            
            if season_est != None:
                duration= (date_ranges[1]-date_ranges[0]).days + 1

                ndvi_season = ndvi_smooth[date_ranges[0]:date_ranges[1]]
                vhvv_season = vhvv_smooth[date_ranges[0]:date_ranges[1]]
                b7b6_season = b7b6_smooth[date_ranges[0]:date_ranges[1]]
                rnfl_season = rainfall[date_ranges[0]:date_ranges[1]]
                tmax_season = tempmax[date_ranges[0]:date_ranges[1]]
                tmin_season = tempmin[date_ranges[0]:date_ranges[1]]
                
# =============================================================================
#                 ndvi_1=chunks(ndvi_season,10)
#                 vhvv_1=chunks(vhvv_season,10)
#                 b7b6_1=chunks(b7b6_season,10)
#                 tmax_1=chunks(tmax_season,10)
#                 tmin_1=chunks(tmin_season,10)
#                 rnfl_1=chunks(rnfl_season,10)
# =============================================================================
                ndvi_final = np.zeros([215])
                vhvv_final = np.zeros([215])
                b7b6_final = np.zeros([215])
                rnfl_final = np.zeros([215])
                tmax_final = np.zeros([215])
                tmin_final = np.zeros([215])
                
                if duration<=215:
                    try:
                        ndvi_final[0:duration]=ndvi_season
                        vhvv_final[0:duration]=vhvv_season
                        b7b6_final[0:duration]=b7b6_season
                        rnfl_final[0:duration]=rnfl_season
                        tmax_final[0:duration]=tmax_season
                        tmin_final[0:duration]=tmin_season
                        
                        #dictvalues = sections(ndvi_1, vhvv_1, b7b6_1, rnfl_1, tmax_1, tmin_1)
                    
                        dictvalues={'NDVI':ndvi_final,'VHVV':vhvv_final,'B7B6':b7b6_final,'RainFall':rnfl_final,'MaxTemp':tmax_final,'MinTemp':tmin_final}
                        
                        df = pd.DataFrame(dictvalues)
                        #df = df.append(dictvalues, ignore_index=True)
                        
                        # append each crop as df columns and asign 0/1
                        value={}
                        for crop in crop_list:
                            if crop == crop_name:
                                value[crop] = 1
                            else:
                                value[crop] = 0
                                
                        df1 = pd.DataFrame(value,index=[0])
                                        
                        seasonvalue={}
                        for season in season_list:
                            if season == season_est:
                                seasonvalue[season]=1
                            else:
                                seasonvalue[season]=0
                                
                        df2 = pd.DataFrame(seasonvalue,index=[0])
                        
                        #info={'Plot_ID':plot_id}
                        
                        #df3 = pd.DataFrame(info,index=[0])
                        
                        df_final=pd.concat([df, df1, df2],axis=1)
                        
                        
                        for crop in crop_list:
                            if crop == crop_name:
                                df_final[crop][0:215] = 1
                            else:
                                df_final[crop][0:215] = 0
                                
                        for season in season_list:
                            if season == season_est:
                                df_final[season][0:215] = 1
                            else:
                                df_final[season][0:215] = 0
                        
                        
                        #df_final['Plot_ID'][0:215]=plot_id
                        
                    
                        


                        #df=df.append(crop_yield, ignore_index=True)
                        
                        #df['Yield'] = crop_yield 
                        
                        #df=df.append(plot_id, ignore_index=True)
                        
                        #df['Plot_ID'] = plot_id
                        
                        #df['Crop_Name'] = crop_name
                        
                        dflist.append(df_final)
                        
                        #dflist.append(df)
                        
                        X[count,:,:]=df_final
                        y[count,:]= crop_yield
                        count=count+1
                    except:
# =============================================================================
#                         if len(ndvi_season)-len(rnfl_season)==9:
#                             duration_length.append([fpo_id,plot_id, date_ranges[0], date_ranges[1], wkt])
#                         else:
#                             print(len(ndvi_season)-len(rnfl_season))
#                             print(plot_id, duration, len(rnfl_season))
# =============================================================================
                        missing= missing+1
                            
                        pass
                
                else:
                    season_length.append([fpo_id,plot_id])
                
                
                
                
                
            else:
                no_seasons.append([fpo_id, plot_id])
        else:
            partial_data.append([fpo_id, plot_id])
    else:
        no_files.append([fpo_id, plot_id])
            
# final X and Y data points
#finaldata = pd.DataFrame(dflist)

#index_=[i for i in range(0,707)]
#finaldata.index=index_

#%%

h5f = h5py.File('C:\\Users\\aravi\\Documents\\Dvara-Internship\\Crop Yield Estimation\\Dataset\\model-series-m1.h5', 'w')
h5f.create_dataset('X', data= X)
h5f.create_dataset('Y', data= y)
h5f.close()


#%%
