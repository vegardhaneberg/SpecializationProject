from time_series_cygnss import get_cygnss_and_era5_time_series
from era5 import scale_values, pretty_dates
from Plot import plot
import datetime
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from geopy.distance import distance


use_scaling = False
# reference_date = datetime.datetime(2020, 1, 1, 0, 0, 0)
# cygnss_path = 'Data/cygnss/first7daysJan2020-28-72-8-85.csv'
cygnss_path = '/Users/vegardhaneberg/Desktop/Processed2020-01-withQFs-35-68-6-88.csv'
era5_path = 'Data/ERA5/era5_rain_sm_india_january_2020'
search_area = {'north': 28, 'south': 8, 'west': 72, 'east': 85}
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2020, 1, 26)

time_series = get_cygnss_and_era5_time_series(cygnss_path, era5_path, search_area, start_date, end_date,
                                              samples_minimum=2)

# Check if we have data
if time_series is None:
    print('Not enough measurements for the specifications you made')
else:
    print('We have results!')

# Print data
ground_truth_ts = time_series[0]
cygnss_ts = time_series[1]
print('Ground Truth:', ground_truth_ts)
print('CYGNSS:', cygnss_ts)

# Scale data
if use_scaling:
    ground_truth_ts_list = scale_values(list(ground_truth_ts.values()))
    cygnss_ts = scale_values(list(cygnss_ts.values()))
else:
    ground_truth_ts_list = list(ground_truth_ts.values())
    cygnss_ts = list(cygnss_ts.values())

# Get pretty dates
dates = pretty_dates(start_date, end_date)


# Plotting

cygnss_df = time_series[2]

plot.plot_nice_time_series(cygnss_df, ground_truth_ts, date_column_name='dates')

# Converting hours to days_index
cygnss_df['days'] = cygnss_df['hours_after_jan_2020'].apply(lambda x: math.floor(x/24))

samples = plot.get_bar_plot_stats(cygnss_df, start_date, end_date)
plot.plot_cygnss_and_number_of_samples(cygnss_df, samples, 'days', 'sr')

# -----------------------------------------------------------------------------
area = time_series[3]

df_lat = cygnss_df.groupby(['dates'], as_index=False)['sp_lat'].mean()
df_long = cygnss_df.groupby(['dates'], as_index=False)['sp_lon'].mean()

overall_df = pd.merge(df_lat, df_long, on=['dates'], how='inner')

point = (np.average([area['north'], area['south']]), np.average([area['west'], area['east']]))

overall_df['distance_from_grid_point'] = overall_df.apply(lambda x: distance(point, (x['sp_lat'], x['sp_lon'])).m, axis=1) # Bytt ut med linjen under
# overall_df['distance_from_grid_point'] = overall_df.apply(lambda x: math.sqrt((x['sp_lat'] - point[0])**2 + (x['sp_lon'] - point[1])**2), axis=1)
overall_df = overall_df.rename(columns={'sp_lon': 'avg_sp_lon', 'sp_lat': 'avg_sp_lat'})

test = pd.merge(cygnss_df, overall_df, on=['dates'], how='left')

test['spread'] = test.apply(lambda x: distance((x['sp_lat'], x['sp_lon']), (x['avg_sp_lat'], x['avg_sp_lon'])).m, axis=1)
# test['spread'] = test.apply(lambda x: math.sqrt((x['sp_lat']-x['avg_sp_lat'])**2 + (x['sp_lon']-x['avg_sp_lon'])**2), axis=1)

test = test.groupby(['dates'], as_index=False)['spread'].mean()

overall_df['spread'] = test['spread']
overall_df = overall_df[['dates', 'distance_from_grid_point', 'spread']]

# -----------------------------------------------------------------------------


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Time Series Statistics')
ax1.bar(dates, list(overall_df['spread']))
ax1.set_ylabel('[m]')
ax1.set_xlabel('Day After 1st of January 2020')
ax1.set_title('Spread')

ax2.bar(dates, list(overall_df['distance_from_grid_point']))
ax2.set_title('Average Distance From Central Point')
ax2.set_ylabel('[m]')
ax2.set_xlabel('Day After 1st of January 2020')


fig.tight_layout(pad=3.0)
plt.subplot_tool()
plt.savefig('statistics.png')
plt.show()

"""
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
plt.subplots_adjust(hspace=0.5)
fig.suptitle('Time Series Statistics')

ax1.bar(dates, list(overall_df['spread']))
ax1.set_title('Spread')

ax2.bar(dates, list(overall_df['distance_from_grid_point']))
ax2.set_title('Average Distance From Central Point')


sns.lineplot(data=cygnss_df, x='dates', y='sr', color="g")
ax3.set_xlabel("Date", fontsize=14)
ax3.set_ylabel("Surface Reflectivity [dB]", color="g", fontsize=14)
ax3.legend(['Surface Reflectivity (SR)', 'std SR'], loc=2)
ax3.grid()

twin_axis = ax3.twinx()
sns.lineplot(x=list(ground_truth_ts.keys()), y=list(ground_truth_ts.values()), color="b", ax=twin_axis)
twin_axis.set_ylabel("Soil Moisture [m]", color="b", fontsize=14)

twin_axis.legend(['Soil Moisture (SM)'], loc=0)

plt.show()
"""
"""
Note to self:
Ønsker å plotte time series fra cygnss, era5 sammen i første linje plot
Plotte cygnss time series og samples i andre plot. Disse ligger lagret i variablen samples.
Neste plot skal være et bar plot med spredning. Disse ligger i overall_df['spread']
Til slutt, avstand fra grid box. Disse ligger i overall_df['distance_from_grid_point']
"""
