import h5py
import pandas as pd
from tqdm import tqdm
import numpy as np
from geopy.distance import distance
import cdsapi
from time import time
import time
import math
import netCDF4 as nc
import datetime
import numpy as np
import pandas as pd
import scipy.interpolate.interpnd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import datetime
from matplotlib import pyplot as plt, figure
from ground_truth import plot
import netCDF4 as nc
from Plot.plot import universal_plot
from TemporalDiffMain import read_and_gridbox_cygnss_data
from matplotlib.colors import LogNorm
from random import random


def plot_corr(x, y, x_label_name, y_label_name, title, log=False):
    if not log:
        m, b = np.polyfit(x, y, 1)
    if log:
        a_log, b_log = np.polyfit(np.log(x), y, 1)  # y = a + b*log(x)

    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=18)
    hist = ax.hist2d(x, y, (70, 70), cmap=plt.cm.jet, norm=LogNorm(), cmin=1)

    if not log:
        line, = plt.plot(x, m * x + b, 'red')
        line.set_label('Linear Fit: ' + str(round(m, 2)) + 'X + ' + str(round(b, 2)))
    if log:
        line, = plt.plot(x, a_log + b_log*np.log(x), 'red')
        line.set_label('Log fit: ' + str(round(a_log, 2)) + ' + ' + str(round(b_log, 2)) + 'Log(x)')

    line.set_linewidth(3)
    ax.legend()

    bar = fig.colorbar(hist[3], ax=ax)
    bar.ax.set_title("Count in log scale")

    ax.set_xlabel(x_label_name, fontsize=12)
    ax.set_ylabel(y_label_name, fontsize=12)

    save_filename = str(math.floor(random()*100000)) + '.png'
    print(save_filename)
    plt.savefig(save_filename)
    plt.show()


def get_smap(month_number: int, printing=False, start_day=1, end_day=31):
    month_dict = {1: 'Jan', 8: 'Aug', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'Mai', 6: 'Jun', 7: 'Jul', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Des'}
    if month_number < 10:
        month_string = '0' + str(month_number)
    else:
        month_string = str(month_number)

    dfs = []
    for i in range(start_day, end_day + 1):
        print('Processing SMAP day ' + str(i) + ' of ' + str(end_day) + '...')
        if month_number == 8 and i == 27:
            path = '/Users/vegardhaneberg/Desktop/SMAP Filtered Aug 2020 nc/SMAP_L3_SM_P_E_20200827_R17000_003_HEGOUT.nc'
        else:
            if i < 10:
                path = '/Users/vegardhaneberg/Desktop/SMAP Filtered ' + month_dict[month_number] + ' 2020 nc/SMAP_L3_SM_P_E_2020' + month_string + '0' + str(i) + '_R17000_001_HEGOUT.nc'
            else:
                path = '/Users/vegardhaneberg/Desktop/SMAP Filtered ' + month_dict[month_number] + ' 2020 nc/SMAP_L3_SM_P_E_2020' + month_string + str(i) + '_R17000_001_HEGOUT.nc'

        ds = nc.Dataset(path)
        sm = ds['Soil_Moisture_Retrieval_Data_AM']

        latitudes = []
        longitudes = []
        moistures = []

        for lat in range(len(sm['latitude'])):
            for long in range(len(sm['longitude'][lat])):
                latitudes.append(sm['latitude'][lat][long])
                longitudes.append(sm['longitude'][lat][long])
                moistures.append(sm['soil_moisture'][lat][long])

        df = pd.DataFrame.from_dict({'lat': latitudes, 'long': longitudes, 'smap_sm': moistures})

        # Filter out missing values
        trimmed_df = df[df['smap_sm'] != -9999.0]

        if len(trimmed_df) > 0:
            dfs.append(trimmed_df)
            if printing:
                print('Number of missing values:', len(df) - len(trimmed_df))
                print('Number of data points with value:', len(trimmed_df))
                index = list(trimmed_df['smap_sm']).index(max(list(trimmed_df['smap_sm'])))
                print("Peak SM value:", list(trimmed_df['smap_sm'])[index])
                print("Peak SM value at: (" + str(list(trimmed_df['lat'])[index]) + ", " + str(list(trimmed_df['long'])[index]) + ")")

    smap_df = pd.DataFrame()
    for df in dfs:
        smap_df = smap_df.append(df)

    # Grid Boxing and Averaging
    smap_df['lat'] = smap_df['lat'].apply(lambda x: round(x, 1))
    smap_df['long'] = smap_df['long'].apply(lambda x: round(x, 1))
    smap_df = smap_df.groupby(['lat', 'long'], as_index=False)['smap_sm'].mean()
    print('--------------------END SMAP---------------------')
    return smap_df


# Parameters
smoothening = 1
cut_off_angle = 60
test_area = {'north': 32.5, 'south': 24.7, 'west': 69.6, 'east': 79.8}
"""
SW: 24.7,69.6
NE: 32.5,79.8
"""

# January Retrieval
smap_jan = get_smap(1)
cygnss_jan, missing_cygnss_jan = read_and_gridbox_cygnss_data('/Users/vegardhaneberg/Desktop/Jan2020-38-45-8-85.csv',
                                                              test_area,
                                                              smooth=smoothening,
                                                              angle_cut_of=cut_off_angle)

# January Correlation
merged_jan = pd.merge(smap_jan, cygnss_jan, on=['lat', 'long'], how='inner')
corr_jan = merged_jan['smap_sm'].astype('float64').corr(merged_jan['avg_sr'].astype('float64'))


# August
smap_aug = get_smap(8)
cygnss_aug, missing_cygnss_aug = read_and_gridbox_cygnss_data('/Users/vegardhaneberg/Desktop/Aug2020-38-45-8-85.csv',
                                                              test_area,
                                                              smooth=smoothening,
                                                              angle_cut_of=cut_off_angle)

vmin_cygnss = min(min(list(cygnss_jan['avg_sr'])), min(list(cygnss_aug['avg_sr'])))
vmax_cygnss = max(max(list(cygnss_jan['avg_sr'])), max(list(cygnss_aug['avg_sr'])))

vmin_smap = min(min(list(smap_jan['smap_sm'])), min(list(smap_aug['smap_sm'])))
vmax_smap = max(max(list(smap_jan['smap_sm'])), max(list(smap_aug['smap_sm'])))

vmin_std = min(min(list(cygnss_jan['std'])), min(list(cygnss_aug['std'])))
vmax_std = max(max(list(cygnss_jan['std'])), max(list(cygnss_aug['std'])))

# January Plot
universal_plot(smap_jan, 'smap_sm', title='SMAP SM January 2020', bar_title='Soil Moisture cm^3/cm^3', save=True, vmin=vmin_smap, vmax=vmax_smap)
universal_plot(cygnss_jan, 'avg_sr', title="CYGNSS SR January 2020", bar_title="Surface Reflectivity [dB]", save=True, vmin=vmin_cygnss, vmax=vmax_cygnss)
universal_plot(cygnss_jan, 'std', title=" CYGNSS STD January 2020", bar_title="STD [dB]", std=True, dot_size=3, vmin=vmin_std, vmax=vmax_std)

# August Plot
universal_plot(smap_aug, 'smap_sm', title='SMAP SM August 2020', bar_title='Soil Moisture cm^3/cm^3', save=True, vmin=vmin_smap, vmax=vmax_smap)
universal_plot(cygnss_aug, 'avg_sr', title='CYGNSS SR August 2020', bar_title='Surface Reflectivity [dB]', save=True, vmin=vmin_cygnss, vmax=vmax_cygnss)
universal_plot(cygnss_aug, 'std', title=" CYGNSS STD August 2020", bar_title="STD [dB]", std=True, dot_size=3,  vmin=vmin_std, vmax=vmax_std)

# August Correlation
merged_aug = pd.merge(smap_aug, cygnss_aug, on=['lat', 'long'], how='inner')
corr_aug = merged_aug['smap_sm'].astype('float64').corr(merged_aug['avg_sr'].astype('float64'))

# Average SR and SM for both months
cygnss_jan_avg = cygnss_jan['avg_sr'].mean()
cygnss_aug_avg = cygnss_aug['avg_sr'].mean()
smap_jan_avg = smap_jan['smap_sm'].mean()
smap_aug_avg = smap_aug['smap_sm'].mean()

# Temporal Difference
cygnss_jan = cygnss_jan.rename(columns={'avg_sr': 'avg_sr_jan'})
cygnss_aug = cygnss_aug.rename(columns={'avg_sr': 'avg_sr_aug'})
smap_jan = smap_jan.rename(columns={'smap_sm': 'smap_sm_jan'})
smap_aug = smap_aug.rename(columns={'smap_sm': 'smap_sm_aug'})

merged_cygnss = pd.merge(cygnss_jan, cygnss_aug, on=['lat', 'long'], how='inner')
merged_smap = pd.merge(smap_jan, smap_aug, on=['lat', 'long'], how='inner')
merged_cygnss['diff_cygnss'] = merged_cygnss['avg_sr_aug'] - merged_cygnss['avg_sr_jan']
merged_smap['diff_smap'] = merged_smap['smap_sm_aug'] - merged_smap['smap_sm_jan']

difference_merged = pd.merge(merged_cygnss, merged_smap, on=['lat', 'long'], how='inner')


# Temporal Difference Correlation
corr_diff = difference_merged['diff_cygnss'].astype('float64').corr(difference_merged['diff_smap'].astype('float64'))

print('Correlaton January:', corr_jan)
print('Correlaton August:', corr_aug)
print('Correlaton Difference:', corr_diff)
print('--------------------------------------------')
print('AVG SR Jan CYGNSS:', cygnss_jan_avg)
print('AVG SR Aug CYGNSS:', cygnss_aug_avg)
print('Difference CYGNSS:', cygnss_jan_avg - cygnss_aug_avg)
print('AVG SM Jan ERA5:', smap_jan_avg)
print('AVG SM Aug ERA5:', smap_aug_avg)
print('Difference ERA5:', smap_jan_avg - smap_aug_avg)

universal_plot(difference_merged, 'diff_cygnss', title='CYGNSS Difference SR Jan-Aug 2020', bar_title='Surface Reflectivity [dB]', save=True)
universal_plot(difference_merged, 'diff_smap', title='SMAP Difference SM Jan-Aug 2020', bar_title='Soil Moisture cm^3/cm^3', save=True)

plot_corr(merged_jan['smap_sm'], merged_jan['avg_sr'], x_label_name='SMAP SM [cm^3/cm^3]', y_label_name='CYGNSS SR [dB]', title='CYGNSS and SMAP Histogram January 2020', log=False)
plot_corr(merged_aug['smap_sm'], merged_aug['avg_sr'], x_label_name='SMAP SM [cm^3/cm^3]', y_label_name='CYGNSS SR [dB]', title='CYGNSS and SMAP Histogram August 2020', log=False)
plot_corr(difference_merged['diff_smap'], difference_merged['diff_cygnss'], x_label_name='SMAP Difference SM [cm^3/cm^3]', y_label_name='CYGNSS Difference SR [dB]', title='CYGNSS and SMAP Histogram Difference', log=False)

