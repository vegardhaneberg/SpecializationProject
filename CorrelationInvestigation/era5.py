"""
To make this file run on mac, do the following:
1. Create an account at https://cds.climate.copernicus.eu/user/register?destination=%2F%23!%2Fhome
2. Log in to CDS at https://cds.climate.copernicus.eu/user/login
3. Copy the url and key at https://cds.climate.copernicus.eu/api-how-to
4. Create the file .cdsapirc at the home directory of your machine
5. Paste the url and key in the .cdsapirc file
6. Accept the terms if you get an error message
"""
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
from Plot import plot


def download_data(area, year, month, day, hour, minute, nc=True):
    """
    Function to download ERA5 dataset containing soil moisture for the area and date/time given as function input.
    In order for this function to work, you must have an account at cds.climate.copernicus and api keys in a .cdsapirc
    file. Follow the instructions above.
    :param area: dict with values for north, west, south, east.
    :param year: the year to download data from. Can be int or string.
    :param month: the month to download data from. Can be int or string.
    :param day: the day to download data from. Can be int or string.
    :param hour: the hour to download data from. Can be int or string.
    :param minute: the minute to download data from. Can be int or string.
    :param nc: boolean. If true the downloaded file will be NetCDF, else GRIB.
    :return: the path to the saved file
    """
    start = time()
    if nc:
        download_format = 'nc'
    else:
        download_format = 'grib'

    # Creating the request dictionary
    request_dict = {
        'format': download_format,
        'variable': 'volumetric_soil_water_layer_1',
        'year': year,
        'month': month,
        'day': day,
        'time': str(hour) + ':' + str(minute),
        'area': [area['north'], area['west'], area['south'], area['east']],
    }

    # Creating the client
    c = cdsapi.Client()
    print('Getting era5 data from this region:\n', area)

    # Posting the request
    save_filename = "Data/" + "download." + download_format
    c.retrieve('reanalysis-era5-land', request_dict, save_filename)

    print('Data retrieved')
    print('Time:', time() - start)

    return save_filename


def convert_lake_sm_values(path, era5_df):
    try:
        ds = xr.open_dataset(path)
    except:
        ds = xr.open_dataset('../' + path)

    df = ds.to_dataframe()
    df = df.reset_index()

    df['lat'] = df['latitude'].apply(lambda x: round(x, 2))
    df['long'] = df['longitude'].apply(lambda x: round(x, 2))
    df = df.drop(['latitude', 'longitude'], axis=1)

    df = df[df['lsm'] < 0.5]

    lake_coordinates = list(zip(df['lat'], df['long']))
    print('start', 'len era5_df:', len(era5_df))
    era5_df['swvl1'] = era5_df.apply(lambda x: 1 if (x['lat'], x['long']) in lake_coordinates else x['swvl1'], axis=1)
    print('end')
    return era5_df


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


def read_and_filter_era5_data(path, date, target_value='swvl1', remove_lake=True):
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

    start = pd.Timestamp(2020, 1, 1)
    time_converted = []
    for t in time:
        diff = t - start
        time_converted.append(diff.days * 24 + diff.seconds / 3600)

    long, lat, time_converted = np.meshgrid(longitude, latitude, time_converted, indexing='ij')
    long = np.array(long).flatten()
    lat = np.array(lat).flatten()
    time_converted = np.array(time_converted).flatten()

    df = pd.DataFrame({'long': long, 'lat': lat, 'time': time_converted, target_value: df[target_value]})
    print(len(df))
    df = df[df['time'] == convert_date_to_hours(start, date)]
    print(len(df))

    if target_value == 'swvl1' and remove_lake:
        df = remove_lakes('Data/ERA5/lsm.nc', df)
    return df


def read_era5_data(path, target_value='swvl1', remove_lake=True):
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

    start = pd.Timestamp(2020, 1, 1)
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
        df = remove_lakes('Data/ERA5/lsm.nc', df)

    return df


def interpolate_hours(era5_path, cygnss_hours, target_value='swvl1', remove_lake=True):
    df = read_era5_data(era5_path, target_value=target_value, remove_lake=remove_lake)

    return_dict = {}
    for hour in cygnss_hours:
        tmp_df = df.copy(deep=True)
        tmp_df = tmp_df[tmp_df['time'] == round(hour, 0)]
        tmp_df = tmp_df.drop(['time'], axis=1)
        tmp_data_points = list(zip(list(tmp_df['long']), list(tmp_df['lat'])))
        tmp_sm = list(tmp_df[target_value])
        tmp_int_func = LinearNDInterpolator(tmp_data_points, tmp_sm)
        return_dict[hour] = tmp_int_func

    return return_dict


def interpolate_one_hour(era5_path: str, hour: int, target_value='swvl1', remove_lake=True) -> LinearNDInterpolator:
    df = read_era5_data(era5_path, target_value=target_value, remove_lake=remove_lake)

    # Nearest neighbour interpolation in time
    df = df[df['time'] == round(hour, 0)]
    df = df.drop(['time'], axis=1)

    # Linear interpolation in spatial domain
    coordinates = list(zip(list(df['long']), list(df['lat'])))
    soil_moisture = list(df[target_value])
    interpolation_function = LinearNDInterpolator(coordinates, soil_moisture)

    return interpolation_function


def convert_date_to_hours(start_date, to_date):
    diff = to_date - start_date
    return round(diff.days * 24 + diff.seconds / 3600, 0)


def filter_era5_location(df, area, include_equals=False):
    if include_equals:
        filtered_df = df[df.lat <= area['north']]
        filtered_df = filtered_df[filtered_df.lat >= area['south']]
        filtered_df = filtered_df[filtered_df.long <= area['east']]
        filtered_df = filtered_df[filtered_df.long >= area['west']]
    else:
        filtered_df = df[df.lat < area['north']]
        filtered_df = filtered_df[filtered_df.lat > area['south']]
        filtered_df = filtered_df[filtered_df.long < area['east']]
        filtered_df = filtered_df[filtered_df.long > area['west']]
    return filtered_df


def get_era5_time_series(path, area, start_date, end_date, target_value='tp', include_equals=False):
    print('Reading era5 data...')
    df = read_era5_data(path, target_value=target_value)
    print('Collected ' + str(len(df)) + ' era5 measurements\n')

    reference_date = datetime.date(2020, 1, 1)
    delta = datetime.timedelta(days=1)
    time_series = {}

    while start_date <= end_date:
        print('Calculating era5 time series for date ' + str(start_date) + '...')
        start_hour = convert_date_to_hours(reference_date, start_date)
        end_hour = start_hour + 24
        current_df = df.copy(deep=True)
        current_df = current_df[current_df['time'] >= start_hour]
        current_df = current_df[current_df['time'] < end_hour]
        current_df = filter_era5_location(current_df, area, include_equals=include_equals)
        total_rain = sum(list(current_df[target_value]))
        samples = len(current_df)

        if samples == 0:
            return None

        avg_rain = total_rain / samples
        time_series[start_date.day] = avg_rain
        start_date += delta
    print()
    return time_series


def pretty_dates(start_date: datetime.date, end_date: datetime.date) -> list:
    delta = datetime.timedelta(days=1)
    pretty_dates_list = []
    while start_date <= end_date:
        pretty_dates_list.append(str(start_date.day))
        start_date += delta
    return pretty_dates_list


def scale_values(values: list) -> list:
    scaled_list = []

    min_value = min(values)
    for i in range(len(values)):
        scaled_list.append(values[i] - min_value)

    max_value = max(scaled_list)
    for i in range(len(scaled_list)):
        scaled_list[i] = scaled_list[i] / max_value

    return scaled_list


def validate_era5_time_series(time_series: dict) -> bool:
    valid = True
    for value in time_series.values():
        if np.isnan(value):
            valid = False
    return valid


def main():
    start_date = datetime.date(2020, 8, 1)
    end_date = datetime.date(2020, 8, 15)
    test_area = {'north': 21.3, 'south': 21.1, 'west': 79.5, 'east': 79.7}
    rain_time_series = get_era5_time_series('../Data/ERA5/era5_rain_sm_india_august_2020',
                                            test_area,
                                            start_date,
                                            end_date,
                                            target_value='tp',
                                            include_equals=True)
    sm_time_series = get_era5_time_series('../Data/ERA5/era5_rain_sm_india_august_2020',
                                          test_area,
                                          start_date,
                                          end_date,
                                          target_value='swvl1',
                                          include_equals=True)

    # Plotting
    dates = pretty_dates(start_date, end_date)
    plt.title('Relative rain and soil moisture')
    plt.ylabel('Percentage rain and soil moisture ')
    plt.xlabel('Date')
    plt.rcParams["figure.figsize"] = (35, 50)
    plt.xticks(rotation=45)
    plt.plot(dates, scale_values(list(rain_time_series.values())))
    plt.plot(dates, scale_values(list(sm_time_series.values())))
    plt.legend(['Rain', 'Soil moisture'])
    plt.show()


def print_era5_area(df):
    area = {'north': max(list(df['lat'])),
            'south': min(list(df['lat'])),
            'west': min(list(df['long'])),
            'east': max(list(df['long']))}
    print(area)
    print('from', min(list(df['time'])) / 24, 'to', max(list(df['time'])) / 24, 'days')


def test():
    not_removed_lakes_df = read_era5_data('../Data/ERA5/kenya_era5_sm.nc', target_value='swvl1', remove_lake=False)
    removed = remove_lakes('../Data/ERA5/lsm.nc', not_removed_lakes_df)

    plot.universal_plot(not_removed_lakes_df, target_value='swvl1')
    plot.universal_plot(removed, target_value='swvl1')


def get_era5_time_series_12h(path: str, area: dict, start: datetime.datetime, end: datetime.datetime, target_value='tp'):
    print('Reading era5 data...')
    df = read_era5_data(path, target_value=target_value)
    print('Collected ' + str(len(df)) + ' era5 measurements')
    print('Filtering on this area:', area)
    df = filter_era5_location(df, area)
    print('Done area filtering\n')

    reference_start_date_time = datetime.datetime(2020, 1, 1, 0, 0, 0)
    delta = datetime.timedelta(hours=12)
    start += delta

    time_series = {}

    while start < end:
        start_hour = convert_date_to_hours(reference_start_date_time, start)
        end_hour = start_hour + 12
        start_hour -= 12

        current_df = df.copy(deep=True)
        current_df = current_df[current_df['time'] >= start_hour]
        current_df = current_df[current_df['time'] < end_hour]
        total_rain = sum(list(current_df[target_value]))
        samples = len(current_df)

        if samples == 0:
            return None

        time_series[start] = total_rain/samples

        print(start_hour, end_hour)
        start += delta

    return time_series


if __name__ == '__main__':
    test()
    """
    a = {'north': 27.2, 'south': 27.0, 'west': 84.4, 'east': 84.6}
    start = datetime.datetime(2020, 1, 1, 0, 0, 0)
    end = datetime.datetime(2020, 1, 7, 0, 0, 0)
    ts = get_era5_time_series_12h('../Data/ERA5/era5_rain_sm_india_january_2020', start, end, a)
    print(ts)
    """



















def validate_era5_dataset(path, use_print=False):
    ds = nc.Dataset(path)
    print(ds.variables)
    print(ds['swvl1'][:, :, :].shape)

    for t in range(len(ds['swvl1'][:, :, :])):
        for lat in range(len(ds['swvl1'][:, :, :][t])):
            for long in range(len(ds['swvl1'][:, :, :][t][lat])):
                if use_print:
                    print('time:', ds['time'][t])
                    print('lat:', ds['latitude'][lat])
                    print('long:', ds['longitude'][long])
                    print(ds['swvl1'][:, :, :][t][lat][long])
                    print('---------------------------------')
                if ds['swvl1'][:, :, :][t][lat][long] > 0.8 or ds['swvl1'][:, :, :][t][lat][long] < 0:
                    print(ds['swvl1'][:, :, :][t][lat][long])
