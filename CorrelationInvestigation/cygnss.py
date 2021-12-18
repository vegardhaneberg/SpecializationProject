from pydap.client import open_url
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time


pd.set_option('display.max_columns', None)


def generate_url(year, month, day, satellite_number):

    day_of_year = datetime(year, month, day).timetuple().tm_yday
    date_string = str(year) + str(month).zfill(2) + str(day).zfill(2)

    base_url = 'https://podaac-opendap.jpl.nasa.gov/opendap/hyrax/allData/cygnss/L1/v3.0/'
    specific_url = str(year) + '/' + str(day_of_year).zfill(3) + '/cyg0' + str(satellite_number) + '.ddmi.s' + \
                   date_string + '-000000-e' + date_string + '-235959.l1.power-brcs.a30.d31.nc'
    data_url = base_url + specific_url
    # clickable_url = base_url + specific_url + '.html'

    return data_url + '?sp_lat,sp_lon,ddm_timestamp_utc,ddm_snr,gps_tx_power_db_w,gps_ant_gain_db_i,rx_to_sp_range,' \
                      'tx_to_sp_range,sp_rx_gain,spacecraft_num'


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def collect_dataset(url, year, month, day, satellite_nr):
    dataset = open_url(url, output_grid=False)

    df = pd.DataFrame()

    for ddm in range(1):  # Remember to change back to 4

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

        ddm_df['ddm_channel'] = np.zeros(len(sp_lon))
        ddm_df['spacecraft_num'] = np.zeros(len(sp_lon))
        ddm_df['year'] = np.zeros(len(sp_lon))
        ddm_df['month'] = np.zeros(len(sp_lon))
        ddm_df['day'] = np.zeros(len(sp_lon))
        ddm_df['sp_lat'] = sp_lat.tolist()
        ddm_df['sp_lon'] = sp_lon.tolist()
        ddm_df = ddm_df.assign(ddm_channel=ddm)
        ddm_df = ddm_df.assign(spacecraft_num=satellite_nr)
        ddm_df = ddm_df.assign(year=year)
        ddm_df = ddm_df.assign(month=month)
        ddm_df = ddm_df.assign(day=day)

        ddm_df['ddm_timestamp_utc'] = ddm_timestamp_utc.tolist()
        ddm_df['ddm_snr'] = ddm_snr.tolist()
        ddm_df['gps_tx_power_db_w'] = gps_tx_power_db_w.tolist()
        ddm_df['gps_ant_gain_db_i'] = gps_ant_gain_db_i.tolist()
        ddm_df['rx_to_sp_range'] = rx_to_sp_range.tolist()
        ddm_df['tx_to_sp_range'] = tx_to_sp_range.tolist()
        ddm_df['sp_rx_gain'] = sp_rx_gain.tolist()

        for col in ddm_df.columns:
            if col != 'ddm_channel' and col != 'ddm_timestamp_utc' and col != 'spacecraft_num':
                ddm_df[col] = ddm_df[col].apply(lambda x: x[0])
        df = df.append(ddm_df, ignore_index=True)

    return df


def collect_data(url):
    data = open_url(url, output_grid=False)
    return data


def calculate_sr(snr, p_r, g_t, g_r, d_ts, d_sr):
    # snr(dB), p_r(dBW), g_t(dBi), g_r(dBi), d_ts(meter), d_sr(meter)
    return snr - (10*np.log10(p_r)) - (10*np.log10(g_t)) - (10*np.log10(g_r)) - (20*np.log10(0.19)) + (20*np.log10(d_ts+d_sr)) + (20*np.log10(4*np.pi))


def plot_smap(df, dot_size=0.5):
    plt.scatter(x=list(df['sp_lon']), y=list(df['sp_lat']), c=list(df['sr']), cmap='Spectral', s=dot_size)
    plt.colorbar()
    plt.title('Surface Reflectivity')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('sr_test.svg')
    plt.show()


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


def filter_location(df, location):
    filtered_df = df[df.sp_lat < location[0]]
    filtered_df = filtered_df[filtered_df.sp_lat > location[2]]
    filtered_df = filtered_df[filtered_df.sp_lon < location[3]]
    filtered_df = filtered_df[filtered_df.sp_lon > location[1]]
    return filtered_df


def compute_surface_reflectivity(df):
    df['sr'] = df.apply(lambda row: calculate_sr(row.ddm_snr, row.gps_tx_power_db_w, row.gps_ant_gain_db_i,
                                                 row.sp_rx_gain, row.tx_to_sp_range, row.rx_to_sp_range), axis=1)
    return df


def get_cygnss_df(year, month, day, location):

    df = pd.DataFrame()

    for i in range(1):  # Remember to change back to 8 satellites for all data collection
        satellite_start = time.time()
        print('Starting computations for satellite number ' + str(i+1) + '...')
        print('------------------------------------------------------------')

        print('Generating url...')
        data_url = generate_url(year, month, day, i+1)

        print('Collecting raw data...')
        raw_data = collect_data(data_url)

        print('Collecting data as a DataFrame...')
        satellite_df = collect_dataset(data_url, year, month, day, i+1)

        print('Removing fill values...')
        rows_before_removal = satellite_df.shape[0]
        satellite_df = remove_fill_values(satellite_df, raw_data)
        rows_after_removal = satellite_df.shape[0]
        print('Removed ' + str(rows_before_removal - rows_after_removal) + ' rows containing fill values...')

        print('Filtering the DataFrame based on provided location...')
        satellite_df = filter_location(satellite_df, location)

        print('Computing surface reflectivity values for all rows...')
        satellite_df = compute_surface_reflectivity(satellite_df)

        df = df.append(satellite_df)
        seconds = time.time()-satellite_start
        print('Collected data for satellite ' + str(i+1) + ' in ' + str(round(seconds/60)) + ' minutes and ' +
              str(seconds % 60) + ' seconds.')
        print('#####################################################')
        print('#####################################################\n\n')

    return df
