import numpy as np
import pandas as pd
import datetime

try:
    from cygnss.cygnss import filter_location
except ModuleNotFoundError:
    from cygnss import filter_location

from ground_truth.era5 import convert_date_to_hours
from ground_truth.era5 import get_era5_time_series, get_era5_time_series_12h


def get_cygnss_time_series_areas(path, area, start_date, end_date, samples_minimum=10):
    df = pd.read_csv(path)

    reference_date = datetime.date(2020, 1, 1)
    first_day_start = convert_date_to_hours(reference_date, start_date)
    first_day_end = first_day_start + 24
    tmp_df = df.copy(deep=True)
    tmp_df = tmp_df[tmp_df['hours_after_jan_2020'] >= first_day_start]
    tmp_df = tmp_df[tmp_df['hours_after_jan_2020'] < first_day_end]

    lat_window = np.arange(area['north'], area['south'], -0.2)
    long_window = np.arange(area['west'], area['east'], 0.2)

    possible_locations = []
    print('Looking for a window with more than ' + str(samples_minimum) + ' samples from cygnss in the first day...')
    for lat in lat_window:
        for long in long_window:
            current_location = {'north': round(lat + 0.2, 2),
                                'south': round(lat, 2),
                                'west': round(long, 2),
                                'east': round(long + 0.2, 2)}
            s = len(filter_location(tmp_df, [lat + 0.2, long, lat, long + 0.2]))
            if s >= samples_minimum:
                possible_locations.append(current_location)

    print(len(possible_locations), 'possible locations after first day')

    # Looping through the rest of the days.
    delta = datetime.timedelta(days=1)
    start_date += delta
    print('Filtering on these days in the cygnss data...')
    while start_date <= end_date:
        tmp_possible_locations = []
        # Filtering data on current day
        current_start_hour = convert_date_to_hours(reference_date, start_date)
        current_end_hour = current_start_hour + 24
        tmp_df = df[df['hours_after_jan_2020'] >= current_start_hour]
        tmp_df = tmp_df[tmp_df['hours_after_jan_2020'] < current_end_hour]
        for current_location in possible_locations:
            s = len(filter_location(tmp_df,
                                    [current_location['north'],
                                     current_location['west'],
                                     current_location['south'],
                                     current_location['east']]))
            if s >= samples_minimum:
                tmp_possible_locations.append(current_location)
        possible_locations = tmp_possible_locations.copy()
        start_date += delta

    print(len(possible_locations), 'possible locations after all days\n')

    return possible_locations


def scale_sr_values(df):
    min_sr = df['sr'].min()
    df['sr'] = df['sr'].apply(lambda x: x - min_sr)
    return df


def get_cygnss_time_series(path, area, start_date, end_date):
    print('Reading cygnss data...')
    df = pd.read_csv(path)
    df = scale_sr_values(df)
    print('Collected ' + str(len(df)) + ' measurements from cygnss\n')

    reference_date = datetime.date(2020, 1, 1)
    delta = datetime.timedelta(days=1)
    time_series = {}
    ts_df = pd.DataFrame()

    while start_date <= end_date:
        print('Calculating cygnss time series for date ' + str(start_date) + '...')
        current_hour_start = convert_date_to_hours(reference_date, start_date)
        current_hour_end = current_hour_start + 24
        tmp_df = df.copy(deep=True)
        tmp_df = tmp_df[tmp_df['hours_after_jan_2020'] >= current_hour_start]
        tmp_df = tmp_df[tmp_df['hours_after_jan_2020'] < current_hour_end]
        tmp_df = filter_location(tmp_df, [area['north'], area['west'], area['south'], area['east']])

        dates_list = [start_date.day]*len(tmp_df)
        tmp_df['dates'] = dates_list
        ts_df = ts_df.append(tmp_df)
        # ts_df = ts_df.reset_index()

        total_sr = sum(list(tmp_df['sr']))
        samples = len(tmp_df)

        if samples == 0:
            return None

        avg_sr = total_sr/samples
        time_series[start_date.day] = avg_sr
        start_date += delta
    print()
    return time_series, ts_df


def get_cygnss_time_series_12h(path, area, start_date, end_date):
    print('Reading cygnss data...')
    df = pd.read_csv(path)
    print('Collected ' + str(len(df)) + ' measurements from cygnss\n')
    ts_df = df.copy(deep=True)
    ts_df = filter_location(ts_df, [area['north'], area['west'], area['south'], area['east']])

    reference_date = datetime.datetime(2020, 1, 1, 0, 0, 0)

    ts_df = ts_df[ts_df['hours_after_jan_2020'] >= convert_date_to_hours(reference_date, start_date)]
    ts_df = ts_df[ts_df['hours_after_jan_2020'] < convert_date_to_hours(reference_date, end_date)]

    delta = datetime.timedelta(hours=12)
    start_date += delta
    time_series = {}

    while start_date <= end_date:
        print('Calculating cygnss time series for date ' + str(start_date) + '...')
        current_hour_start = convert_date_to_hours(reference_date, start_date)
        current_hour_end = current_hour_start + 12
        current_hour_start -= 12

        tmp_df = df.copy(deep=True)
        tmp_df = tmp_df[tmp_df['hours_after_jan_2020'] >= current_hour_start]
        tmp_df = tmp_df[tmp_df['hours_after_jan_2020'] < current_hour_end]
        tmp_df = filter_location(tmp_df, [area['north'], area['west'], area['south'], area['east']])

        dates_list = [start_date]*len(tmp_df)
        tmp_df['dates'] = dates_list

        total_sr = sum(list(tmp_df['sr']))
        samples = len(tmp_df)

        if samples == 0:
            return None

        avg_sr = total_sr/samples
        time_series[start_date] = avg_sr
        start_date += delta
    print()
    return time_series, ts_df


def get_cygnss_and_era5_time_series(cygnss_path, era5_path, area, start_date, end_date, samples_minimum=10):
    if start_date.month != 1:
        print('####################')
        print('WARNING!!!')
        print('REMEMBER TO CHANGE START DATE IN the read_era5_data() function')
        print('####################')

    print('Searching for areas in cygnss data...')
    areas = get_cygnss_time_series_areas(cygnss_path, area, start_date, end_date, samples_minimum=samples_minimum)

    # Selecting the first area until we find an area that contains era5 rain data as well. This can happen because
    # the cygnss data is over water as well.

    print('Collecting time series from era5...')
    empty_era5_data = True
    index = 0
    ground_truth_ts = None
    while empty_era5_data:
        ground_truth_ts = get_era5_time_series(era5_path, areas[index], start_date, end_date, target_value='swvl1',
                                               include_equals=False)
        if ground_truth_ts is not None:
            empty_era5_data = False
        else:
            index += 1
            print('No era5 data for the following area: area')
            print(areas[index])
            print('trying again...')
        if (index == len(areas) - 1) and empty_era5_data:
            return None

    # Collecting the cygnss data for the location
    print('FOUND MATCHING AREA:')
    print(areas[index])
    cygnss_ts, cygnss_df = get_cygnss_time_series(cygnss_path, areas[index], start_date, end_date)
    return ground_truth_ts, cygnss_ts, cygnss_df, areas[index]


def cygnss_time_series_one_day(path, area, date):
    df = pd.read_csv(path)

    reference_date = datetime.date(2020, 1, 1)

    hour_start = convert_date_to_hours(reference_date, date)
    hour_end = hour_start + 24
    df = df[df['hours_after_jan_2020'] >= hour_start]
    df = df[df['hours_after_jan_2020'] < hour_end]
    df = filter_location(df, [area['north'], area['west'], area['south'], area['east']])
    total_sr = sum(list(df['sr']))
    samples = len(df)

    if samples == 0:
        print('NO VALUES FOR THIS AREA')
    else:
        avg_sr = total_sr / samples
        print(df.head())
        print('Samples:', len(df))
        print('Total SR:', total_sr)
        print('Avg SR:', avg_sr)
        print('-----------------------------')






