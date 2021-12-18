from time_series_cygnss import get_cygnss_and_era5_time_series
from era5 import scale_values, pretty_dates
from Plot import plot
import datetime
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from cygnss.time_series_cygnss import get_cygnss_time_series, get_era5_time_series


cygnss_path1 = '/Users/vegardhaneberg/Desktop/Processed2020-01-withQFs-35-68-6-88.csv'
# cygnss_path2 = '/Users/vegardhaneberg/Desktop/Processed2020-02-withQFs-35-68-6-88.csv'
cygnss_paths = [cygnss_path1]
era5_path = 'Data/ERA5/era5_rain_sm_india_january_2020'
area = {'north': 28, 'south': 8, 'west': 72, 'east': 85}
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 1, 20)


print('Reading cygnss data...')
cygnss_df = pd.DataFrame()
for path in cygnss_paths:
    cygnss_df = cygnss_df.append(pd.read_csv(path))
print('Collected ' + str(len(cygnss_df)) + ' cygnss measurements\n')

cygnss_df = cygnss_df.rename(columns={'sp_lat': 'lat', 'sp_lon': 'long'})
print('Columns in cygnss_df:', list(cygnss_df.columns))
cygnss_df = cygnss_df[['lat', 'long', 'day_of_year', 'sr']]


print(len(cygnss_df))
step = 0.2
to_bin = lambda x: np.floor(x / step) * step
cygnss_df["latBin"] = cygnss_df.lat.map(to_bin)
cygnss_df["lonBin"] = cygnss_df.long.map(to_bin)
cygnss_df = cygnss_df.groupby(["latBin", "lonBin"])
print(len(cygnss_df))


cygnss_df_sr = cygnss_df.groupby(['latBin', 'lonBin', 'day_of_year'], as_index=False)['sr'].mean()

cygnss_dict_sr = {}

lat_window = np.arange(area['north'], area['south'], -0.1)
long_window = np.arange(area['west'], area['east'], 0.1)

for i in tqdm(range(10)):
    lat = round(lat_window[i], 1)
    for j in range(len(long_window)):
        long = round(long_window[j], 1)
        current_sr = list(cygnss_df_sr[(cygnss_df_sr['lat'] == lat) & (cygnss_df_sr['long'] == long)]['sr'])
        current_days = list(cygnss_df_sr[(cygnss_df_sr['lat'] == lat) & (cygnss_df_sr['long'] == long)]['day_of_year'])
        cygnss_dict_sr[lat, long] = [current_sr, current_days]

longest = 0
best_coord = None

for coord in cygnss_dict_sr.keys():
    measurements = len(cygnss_dict_sr[coord][0])
    if measurements > longest:
        longest = measurements
        best_coord = coord

print(best_coord)

cygnss_df = cygnss_df[(cygnss_df['lat'] == best_coord[0]) & (cygnss_df['long'] == best_coord[1])]



plot.plot_nice_time_series(cygnss_df, ground_truth_ts, date_column_name='day_of_year')





























