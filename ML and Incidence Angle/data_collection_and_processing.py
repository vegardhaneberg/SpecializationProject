from pydap.client import open_url
from datetime import datetime
from calendar import monthrange, month_name

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr

import time
import pickle
import cdsapi
import math
from Plot import incidence_and_ml_plot

# Interpolation
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
import scipy.interpolate.interpnd

# Plotting
from matplotlib import pyplot as plt, figure
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LogNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from plotly import express as px
import cartopy.crs as ccrs

# Set data frame options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#################################
### DATA COLLECTION FUNCTIONS ###
#################################

def generate_url(year, month, day, satellite_number):
    day_of_year = datetime(year, month, day).timetuple().tm_yday
    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/cygnss/L1/v3.0/'
    specific_url = str(year) + '/' + str(day_of_year).zfill(3) + '/cyg0' + str(satellite_number) + '.ddmi.s' + \
                   date_string + '-000000-e' + date_string + '-235959.l1.power-brcs.a30.d31.nc'
    data_url = base_url + specific_url

    return data_url + '?sp_lat,sp_lon,ddm_timestamp_utc,ddm_snr,gps_tx_power_db_w,gps_ant_gain_db_i,rx_to_sp_range,' \
                      'tx_to_sp_range,sp_rx_gain,spacecraft_num,prn_code,track_id,quality_flags,quality_flags_2,sp_inc_angle', day_of_year


def collect_dataset(day_of_year, url, satellite_nr):
    dataset = open_url(url, output_grid=False)
    df = pd.DataFrame()
    track_list = []

    for ddm in range(4):

        ddm_df = pd.DataFrame()
        print("ddm: " + str(ddm))
        sp_lat = np.array(dataset.sp_lat[:, ddm])
        sp_lon = np.array(dataset.sp_lon[:, ddm])
        a, b = (np.where(sp_lon > 180))
        sp_lon[a] -= 360

        ddm_timestamp_utc = np.array(dataset.ddm_timestamp_utc[:, ddm])
        ddm_snr = np.array(dataset.ddm_snr[:, ddm])
        gps_tx_power_db_w = np.array(dataset.gps_tx_power_db_w[:, ddm])
        gps_ant_gain_db_i = np.array(dataset.gps_ant_gain_db_i[:, ddm])
        rx_to_sp_range = np.array(dataset.rx_to_sp_range[:, ddm])
        tx_to_sp_range = np.array(dataset.tx_to_sp_range[:, ddm])
        sp_rx_gain = np.array(dataset.sp_rx_gain[:, ddm])
        track_id = np.array(dataset.track_id[:, ddm])
        prn_code = np.array(dataset.prn_code[:, ddm])
        quality_flags = np.array(dataset.quality_flags[:, ddm])
        quality_flags_2 = np.array(dataset.quality_flags_2[:, ddm])
        sp_inc_angle = np.array(dataset.sp_inc_angle[:, ddm])

        ddm_df['ddm_channel'] = np.zeros(len(sp_lon))
        ddm_df['spacecraft_num'] = np.zeros(len(sp_lon))
        ddm_df['day_of_year'] = np.zeros(len(sp_lon))
        ddm_df['sp_lat'] = sp_lat.tolist()
        ddm_df['sp_lon'] = sp_lon.tolist()
        ddm_df = ddm_df.assign(ddm_channel=ddm)
        ddm_df = ddm_df.assign(spacecraft_num=satellite_nr)
        ddm_df = ddm_df.assign(day_of_year=day_of_year)

        ddm_df['ddm_timestamp_utc'] = ddm_timestamp_utc.tolist()
        ddm_df['ddm_snr'] = ddm_snr.tolist()
        ddm_df['gps_tx_power_db_w'] = gps_tx_power_db_w.tolist()
        ddm_df['gps_ant_gain_db_i'] = gps_ant_gain_db_i.tolist()
        ddm_df['rx_to_sp_range'] = rx_to_sp_range.tolist()
        ddm_df['tx_to_sp_range'] = tx_to_sp_range.tolist()
        ddm_df['sp_rx_gain'] = sp_rx_gain.tolist()
        ddm_df['track_id'] = track_id.tolist()
        ddm_df['prn_code'] = prn_code.tolist()
        ddm_df['sp_inc_angle'] = sp_inc_angle.tolist()
        ddm_df['quality_flags'] = quality_flags.tolist()
        ddm_df['quality_flags_2'] = quality_flags_2.tolist()

        for col in ddm_df.columns:
            if col != 'ddm_channel' and col != 'ddm_timestamp_utc' and col != 'spacecraft_num' and col != 'day_of_year':
                ddm_df[col] = ddm_df[col].apply(lambda x: x[0])
        df = df.append(ddm_df, ignore_index=True)

    return df


def collect_data(url):
    data = open_url(url, output_grid=False)
    return data


def create_cygnss_df(year, month, day):
    df = pd.DataFrame()
    raw_data_list = []
    failed_satellites = []
    
    sat_counter = 0
    failed_attempts = 0

    while sat_counter < 8: 
        try:
            satellite_start = time.time()
            print('Starting computations for satellite number ' + str(sat_counter+1) + '...')
            print('------------------------------------------------------------')

            print('Generating url...')
            data_url, day_of_year = generate_url(year, month, day, sat_counter+1)

            print('Collecting data as a DataFrame...')
            satellite_df = collect_dataset(day_of_year, data_url, sat_counter+1)

            print('Collecting raw data...')
            raw_data = collect_data(data_url)
            raw_data_list.append(raw_data)

            seconds = time.time()-satellite_start
            print('Collected data for satellite ' + str(sat_counter+1) + ' in ' + str(round(seconds/60)) + ' minutes and ' + 
                  str(seconds % 60) + ' seconds.')
            print('#####################################################')
            print('#####################################################\n\n')

            df = df.append(satellite_df)
            sat_counter += 1
        except:
            print('Data collection failed. Trying again...')
            failed_attempts += 1
        
        if failed_attempts == 50:
            failed_satellites.append(sat_counter+1)
            sat_counter += 1
            failed_attempts = 0
            print('Data collection aborted. Trying the next satellite!')
            
    return df, raw_data_list, failed_satellites

############################################
### DATA PROCESSING INDIVIDUAL FUNCTIONS ###
############################################

def calculate_sr_value(snr, p_r, g_t, g_r, d_ts, d_sr):
    # snr(dB), p_r(dBW), g_t(dBi), g_r(dBi), d_ts(meter), d_sr(meter)
    return snr - p_r - g_t - g_r - (20 * np.log10(0.19)) + (20 * np.log10(d_ts + d_sr)) + (20 * np.log10(4 * np.pi))


def compute_surface_reflectivity(df):
    df['sr'] = df.apply(
        lambda row: calculate_sr_value(row.ddm_snr, row.gps_tx_power_db_w, row.gps_ant_gain_db_i, row.sp_rx_gain,
                                       row.tx_to_sp_range, row.rx_to_sp_range), axis=1)
    return df


def calculate_hours_after_jan_value(day_of_year, ddm_timestamp):
    return (day_of_year - 1) * 24 + round(ddm_timestamp / (60 * 60))


def compute_hours_after_jan(df):
    df['hours_after_jan_2020'] = df.apply(
        lambda row: calculate_hours_after_jan_value(row.day_of_year, row.ddm_timestamp_utc), axis=1)
    return df


def generate_unique_track_id_value(track_id, day_of_year, prn_nr, sat_nr):
    return track_id * 10000 + prn_nr * 10 + sat_nr + day_of_year/1000


def compute_unique_track_ids(df):
    df['unique_track_id'] = df.apply(
        lambda row: generate_unique_track_id_value(row.track_id, row.day_of_year, row.prn_code, row.spacecraft_num), axis=1)
    return df


def generate_qf_list(qf_number):
    qf_list = []
    binary = format(qf_number, 'b')
    for i in range(len(binary)):
        if binary[i] == '1':
            qf_list.append(2 ** (int(i)))

    return qf_list


def compute_prn_to_block_value(prn_code):
    iir_list = [2, 13, 16, 19, 20, 21, 22]
    iif_list = [1, 3, 6, 8, 9, 10, 25, 26, 27, 30, 32]
    iir_m_list = [5, 7, 12, 15, 17, 29, 31]
    iii_list = [4, 11, 14, 18, 23, 24]
    
    if prn_code in iir_list:
        return 'IIR'
    elif prn_code in iif_list:
        return 'IIF'
    elif prn_code in iir_m_list:
        return 'IIR-M'
    elif prn_code in iii_list:
        return 'III'
    else:
        return 'UNKNOWN'


def compute_block_code(df):
    df['block_code'] = df.apply(lambda row: compute_prn_to_block_value(row.prn_code), axis=1)
    return df


def compute_daily_hour_column(df):
    df['daily_hour'] = df.apply(lambda row: round(row.ddm_timestamp_utc / (60*60)), axis=1)
    return df


def compute_time_of_day_value(time):
    if time >= 22:
        return 'N'
    elif time >= 16:
        return 'A'
    elif time >= 10:
        return 'D'
    elif time >= 4:
        return 'M'
    else:
        return 'N'
    

def compute_time_of_day(df):
    df['time_of_day'] = df.apply(lambda row: compute_time_of_day_value(row.daily_hour), axis=1)
    return df


def scale_sr_values(df):
    min_sr = df['sr'].min()
    df['sr'] = df['sr'].apply(lambda x: x - min_sr)
    return df


def filter_location(df, location):
    filtered_df = df[df.sp_lat < location[0]]
    filtered_df = filtered_df[filtered_df.sp_lat > location[2]]
    filtered_df = filtered_df[filtered_df.sp_lon < location[3]]
    filtered_df = filtered_df[filtered_df.sp_lon > location[1]]
    return filtered_df


def filter_quality_flags_1(df):
    df['qf_ok'] = df.apply(
        lambda row: (2 or 4 or 5 or 8 or 16 or 17) not in generate_qf_list(int(row.quality_flags)), axis=1)
    df = df[df['qf_ok']]
    return df


def filter_quality_flags_2(df):
    res_df = df
    res_df['qf2_ok'] = res_df.apply(
        lambda row: (1 or 2) not in generate_qf_list(int(row.quality_flags_2)), axis=1)  # Remember to check which qfs
    res_df = res_df[res_df['qf2_ok']]
    return res_df


def remove_fill_values(df, raw_data):
    keys = list(raw_data.keys())
    keys.remove('ddm_timestamp_utc')
    keys.remove('spacecraft_num')
    filtered_df = df

    # Remove rows containing fill values
    for k in keys:
        key = raw_data[k]
        fv = key._FillValue
        filtered_df = filtered_df[filtered_df[k] != fv]

    return filtered_df

#################################
### DATA PROCESSING FUNCTIONS ###
#################################

def raw_df_processing(df, location, qf1_removal=False, qf2_removal=False):
    res_df = df

    print('Filtering the DataFrame based on provided location...')
    res_df = filter_location(res_df, location)

    if qf1_removal:
        print('Removing bad quality values...')
        rows_before_removal = res_df.shape[0]
        res_df = filter_quality_flags_1(res_df)
        rows_after_removal = res_df.shape[0]
        print('Removed ' + str(rows_before_removal - rows_after_removal) + ' rows of bad overall quality...')

    if qf2_removal:
        print('Removing more bad quality values...')
        rows_before_removal = res_df.shape[0]
        res_df = filter_quality_flags_2(res_df)
        rows_after_removal = res_df.shape[0]
        print('Removed ' + str(rows_before_removal - rows_after_removal) + ' rows of bad overall quality...')

    print('Computing surface reflectivity values for all rows...')
    res_df = compute_surface_reflectivity(res_df)

    print('Adding column displaying hours after January 1st 2020...')
    res_df = compute_hours_after_jan(res_df)

    print('Computing unique track ids for all rows...')
    res_df = compute_unique_track_ids(res_df)

    return res_df


def process_monthly_df(retrieval_folder, year, month, location, qf1_removal=True, qf2_removal=False):
    monthly_df = pd.DataFrame()
    num_of_days = monthrange(year, month)[1]

    for i in range(num_of_days):
        csv_path = retrieval_folder + str(month).zfill(2) + '/raw_main_df_' + str(year) + '_' + str(month).zfill(2) + '_' + \
                   str(i + 1) + 'of' + str(num_of_days) + '.csv'
        print('#######################################')
        print('Collecting csv file number ' + str(i + 1) + ' of ' + str(num_of_days) + '...')
        daily_df = pd.read_csv(csv_path)
        print('***Processing the data***')
        daily_df = raw_df_processing(daily_df, location, qf1_removal, qf2_removal)
        monthly_df = monthly_df.append(daily_df, ignore_index=True)
        print('#######################################\n')
    return monthly_df


# Returning the same df if no specific features are selected
def select_df_features(df, feature_list):
    if len(feature_list) > 0:
        return df[feature_list]
    else:
        return df


def store_df_as_csv(df, storage_path):
    df.to_csv(storage_path, index=False)


###############################
### INTERPOLATION FUNCTIONS ###
###############################

def interpolate_ml(df: pd.DataFrame, target_value='swvl1') -> LinearNDInterpolator:
    coordinates = list(zip(list(df['time']), list(df['lat']), list(df['long'])))
    target = df[target_value]
    interpolation_function = LinearNDInterpolator(coordinates, target)
    return interpolation_function


def remove_lakes(path, era5_df):
    try:
        ds = xr.open_dataset(path)
    except:
        ds = xr.open_dataset('../' + path)
    df = ds.to_dataframe()
    df = df.reset_index()

    if 'time' in df.columns:
        df = df.drop(['time'], axis=1)

    df['lat'] = df['latitude'].apply(lambda x: round(x, 2))
    df['long'] = df['longitude'].apply(lambda x: round(x, 2))
    df = df.drop(['latitude', 'longitude'], axis=1)

    df = df[df['lsm'] > 0.5]

    len_before = len(era5_df)
    filtered_df = pd.merge(era5_df, df, how='left', left_on=['lat', 'long'], right_on=['lat', 'long'])
    filtered_df = filtered_df.dropna()
    print('Removed lake items:', len_before - len(filtered_df))
    return filtered_df


def read_era5_data(path, year=2020, target_value='swvl1', remove_lake=True):
    try:
        ds = xr.open_dataset(path)
    except:
        ds = xr.open_dataset('../' + path)
    df = ds.to_dataframe()
    if 'src' in df.columns:
        df = df.drop(['src'], axis=1)
    if 'swvl2' in df.columns:
        df = df.drop(['swvl2'], axis=1)

    longitude = df.index.levels[0]
    latitude = df.index.levels[1]
    time = df.index.levels[2]

    start = pd.Timestamp(year, 1, 1)
    time_converted = []
    for t in time:
        diff = t - start
        time_converted.append(diff.days * 24 + diff.seconds / 3600)

    long, lat, time_converted = np.meshgrid(longitude, latitude, time_converted, indexing='ij')
    long = np.array(long).flatten()
    lat = np.array(lat).flatten()
    time_converted = np.array(time_converted).flatten()

    df = pd.DataFrame({'long': long, 'lat': lat, 'time': time_converted, target_value: df[target_value]})

    df['lat'] = df['lat'].apply(lambda lat: round(lat, 1))
    df['long'] = df['long'].apply(lambda long: round(long, 1))

    if target_value == 'swvl1' and remove_lake:
        df = remove_lakes('/Volumes/DACOTA HDD/Semester Project CSV/ERA5/lsm.nc', df)
    return df


def get_era5_monthly_df(era5_df, month, year):
    start_day_of_month = datetime(year, month, 1).timetuple().tm_yday
    if month != 12:
        end_day_of_month =  datetime(year, month + 1, 1).timetuple().tm_yday
    else:
        end_day_of_month = datetime(year, 12, 31).timetuple().tm_yday + 1
    
    res_df = era5_df[era5_df.time >= (start_day_of_month-1)*24]
    return res_df[res_df.time < (end_day_of_month-1)*24]


def generate_monthly_interpolation_function(era5, location, month, year, storage_folder):
    location_string = str(location[0]) + '-' + str(location[1]) + '-' + str(location[2]) + '-' + str(location[3])
    monthly_era5_df = get_era5_monthly_df(era5, month, year)
        
    interpolation_start = time.time()
    print('Generating an interpolation function for the month of ' + month_name[month] + '...')
    monthly_interpolation_function = interpolate_ml(monthly_era5_df)
    print('Seconds used to create the interpolation function: ', (time.time() - interpolation_start))
    return monthly_interpolation_function


def load_monthly_interpolation_function(location, month, year, storage_folder):
    location_string = str(location[0]) + '-' + str(location[1]) + '-' + str(location[2]) + '-' + str(location[3])
    
    with open(storage_folder + '/interpolation_function_' + str(year) + '_' + str(month).zfill(2) + '_' + location_string + '.pickle', 'rb') as input_interpolation_file:
        monthly_interpolation_function = pickle.load(input_interpolation_file)
    
    return monthly_interpolation_function


def interpolate_df(df, interpolation_function):
    df['sm'] = df.apply(lambda row: interpolation_function(row.hours_after_jan_2020, row.sp_lat, row.sp_lon), axis=1)
    return df


def generate_monthly_interpolated_cygnss_df(location, month, year, cygnss_storage_folder, interpolation_function):
    location_string = str(location[0]) + '-' + str(location[1]) + '-' + str(location[2]) + '-' + str(location[3])
    cygnss_df = pd.read_csv(cygnss_storage_folder + '/Processed' + str(year) + '-' + str(month).zfill(2) + '-withQFs-' + 
                            location_string + '.csv')
    interpolation_start = time.time()
    print('Interpolating the df for the month of ' + month_name[month] + '...')
    interpolated_df = interpolate_df(cygnss_df, interpolation_function)
    print('Seconds used to interpolate values for soil moisture: ', (time.time() - interpolation_start))
    return interpolated_df
    

def load_monthly_interpolated_df(location, month, year, interpolated_df_storage_folder):
    location_string = str(location[0]) + '-' + str(location[1]) + '-' + str(location[2]) + '-' + str(location[3])
    
    return pd.read_csv(interpolated_df_storage_folder + '/df_with_interpolated_sm_' + str(year) + '_' + str(single_month).zfill(2) + '_' + location_string + '.csv')


def load_periodic_interpolated_df(location, start_month, end_month, year, interpolated_df_storage_folder):
    location_string = str(location[0]) + '-' + str(location[1]) + '-' + str(location[2]) + '-' + str(location[3])
    res_df = pd.DataFrame()
    for i in range(start_month, end_month + 1):
        tmp_df = pd.read_csv(interpolated_df_storage_folder + '/df_with_interpolated_sm_' + str(year) + 
                             '_' + str(i).zfill(2) + '_' + location_string + '.csv')
        res_df = res_df.append(tmp_df, ignore_index=True)
    return res_df


def filter_nan_era5(df):
    try:
        df['sm'] = df['sm'].apply(lambda x: x.item(0))
    except:
        print('SM value was already of type: float')
    df = df.dropna()
    return df


def filter_nan_smap(df):
    try:
        df['smap_sm'] = df['smap_sm'].apply(lambda x: x.item(0))
    except:
        print('SMAP_SM value was already of type: float')
    df = df.dropna()
    return df


def get_smap(path: str, printing=False):

    ds = nc.Dataset(path)
    sm = ds['Soil_Moisture_Retrieval_Data_AM']

    latitudes = []
    longitudes = []
    moistures = []
    times = []

    for lat in range(len(sm['latitude'])):
        for long in range(len(sm['longitude'][lat])):
            latitudes.append(sm['latitude'][lat][long])
            longitudes.append(sm['longitude'][lat][long])
            moistures.append(sm['soil_moisture'][lat][long])
            times.append(sm['tb_time_utc'][lat][long])

    df = pd.DataFrame.from_dict({'lat': latitudes, 'long': longitudes, 'time': times, 'smap_sm': moistures})

    # Filter out missing values
    smap_df = df[df['smap_sm'] != -9999.0]

    if len(smap_df) > 0 and printing:
        print('Number of missing values:', len(df) - len(smap_df))
        print('Number of data points with value:', len(smap_df))
        index = list(smap_df['smap_sm']).index(max(list(smap_df['smap_sm'])))
        print("Peak SM value:", list(smap_df['smap_sm'])[index])
        print("Peak SM value at: (" + str(list(smap_df['lat'])[index]) + ", " + str(list(smap_df['long'])[index]) + ")")

    return smap_df


def conv(t):
    try:
        return pd.Timestamp(t)
    except:
        return pd.Timestamp(t.split('.')[0] + '.000Z')


def convert_time(df: pd.DataFrame) -> pd.DataFrame:
    ref_date = pd.Timestamp('2020-01-01T00:00:00.000Z')

    df['time'] = df['time'].apply(lambda t: conv(t))
    df['time'] = df['time'].apply(lambda t: (t - ref_date).days * 24 + (t - ref_date).seconds / 3600)
    return df


def get_smap_df_year(root_dir: str, year: int, convert_time_hours=True) -> pd.DataFrame:
    first = True
    all_paths = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if not first:
                all_paths.append(os.path.join(subdir, file))
            else:
                first = False

    smap_df = pd.DataFrame()

    for path in all_paths:
        path_split = path.split('_')
        print(path_split)
        if len(path_split) > 6:
            current_year = int(path_split[4][:4])

            if current_year == year:
                current_df = get_smap(path)
                smap_df = smap_df.append(current_df)

    if convert_time_hours:
        smap_df = convert_time(smap_df)

    return smap_df

