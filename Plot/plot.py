import datetime

from matplotlib import pyplot as plt, rcParams
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from ground_truth import era5
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib


def universal_plot(df, target_value='swvl1', title=None, bar_title=None, vmin=None, vmax=None, save=True, dot_size=0.5, std=False):

    # Settings for the plot
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    lat_list, long_list = get_plot_ticks(df['lat'], df['long'])
    ax.set_xticks(long_list, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_list, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    if std:
        # cmap = matplotlib.cm.get_cmap("magma", 10)
        cmap = 'Greys'
    else:
        cmap = 'Spectral'

    if vmin is not None:
        plt.scatter(df['long'], df['lat'], c=list(df[target_value]), s=dot_size, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.scatter(df['long'], df['lat'], c=list(df[target_value]), s=dot_size, cmap=cmap)
    bar = plt.colorbar()
    if title is None:
        if target_value == 'tp':
            bar.ax.set_title('Rainfall [m]')
            ax.suptitle('ERA5 Rainfall', fontsize=18)
        else:
            bar.ax.set_title('Soil Moisture')
            ax.suptitle('ERA5 Soil Moisture', fontsize=18)
    else:
        bar.ax.set_title(bar_title)
        plt.title(title)
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)

    if save:
        if ('avg_sr' in df.columns or 'smap_sm' in df.columns) and title is not None:
            plt.savefig('../Results/temporal/' + title + '.png')
        if 'time' not in df.columns:
            plt.savefig('../Results/' + title + '.png')
        else:
            if target_value == 'tp':
                plt.savefig('../Results/rain/' + str(target_value) + '_' + str(int(list(df['time'])[0])) + '.png')
            else:
                plt.savefig('../Results/sm/' + str(target_value) + '_' + str(int(list(df['time'])[0])) + '.png')
    plt.show()


def get_plot_ticks(lat_values, long_values):
    min_lat = min(lat_values)
    max_lat = max(lat_values)
    min_long = min(long_values)
    max_long = max(long_values)

    lat_step_size = (max_lat - min_lat) / 3
    long_step_size = (max_long - min_long) / 3

    long_list = [min_long, min_long + long_step_size, min_long + 2 * long_step_size, max_long]
    lat_list = [min_lat, min_lat + lat_step_size, min_lat + 2 * lat_step_size, max_lat]

    # Rounding to two decimals
    long_list = [round(num, 2) for num in long_list]
    lat_list = [round(num, 2) for num in lat_list]

    return lat_list, long_list


def plot_era5_data(path, date, target_value='swvl1', remove_lake=True, boundary_value=None):
    print('Reading data...')
    df = era5.read_and_filter_era5_data(path, date, target_value=target_value, remove_lake=remove_lake)
    # df = era5.read_era5_data(path, target_value=target_value, remove_lake=remove_lake)
    print('Done reading data\n')
    hour = era5.convert_date_to_hours(datetime.date(2020, 1, 1), date)
    print('Filtering on hour...')
    df = df[df['time'] == hour]
    print('Hour filtering done\n')
    if boundary_value is not None:
        print('Filtering values above ' + str(boundary_value) + '...')
        len_before = len(df)
        df = df[df[target_value] > boundary_value]
        print('Filtering done. Removed ' + str(len_before - len(df)) + ' values.')
    print('Starting plotting... We have ' + str(len(df)) + ' data pints')
    universal_plot(df, target_value=target_value)


def plot_scatter_heatmap(df, x_axis_name='sr', y_axis_name='swvl1') -> None:
    fig = px.density_heatmap(df, x=x_axis_name, y=y_axis_name)
    fig.show()


def plot_time_series(dates: list, time_series: list[list], legends: list[str], title='Time Series Surface Reflectivity and Rain'):
    plt.title(title)
    plt.ylabel('Percentage surface reflectivity and rain')
    plt.xlabel('Date')
    plt.rcParams["figure.figsize"] = (35, 50)
    plt.xticks(rotation=20)
    for tm in time_series:
        plt.plot(dates, tm)
    plt.legend(legends)
    plt.show()


def plot_nice_time_series(cygnss_df: pd.DataFrame, era5_time_series: dict, date_column_name='time', sr_column_name='sr',
                          sm_column_name='sm') -> None:
    rcParams['figure.figsize'] = 10, 6
    fig, ax = plt.subplots()
    sns.lineplot(data=cygnss_df, x=date_column_name, y=sr_column_name, color="g")
    ax.set_xlabel("Day After 1st of January 2020", fontsize=14)
    ax.set_ylabel("Surface Reflectivity [dB]", color="g", fontsize=14)
    ax.legend(['Surface Reflectivity (SR)', 'std SR'], loc=2)
    ax.grid()

    twin_axis = plt.twinx()
    sns.lineplot(x=list(era5_time_series.keys()), y=list(era5_time_series.values()), color="b", ax=twin_axis)
    twin_axis.set_ylabel("Soil Moisture [m]", color="b", fontsize=14)

    twin_axis.legend(['Soil Moisture (SM)'], loc=1)
    plt.tight_layout()
    try:
        plt.savefig('nice_time_series.png')
    except:
        plt.savefig('nice_time_series.png')
    plt.show()


def plot_cygnss_and_number_of_samples(cygnss_df: pd.DataFrame, cygnss_samples: dict, date_column_name='time',
                                      sr_column_name='sr') -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_title('Cygnss Times Series', fontsize=16)
    ax1.set_xlabel('Day After 1st of January 2020', fontsize=16)
    ax1.set_ylabel('Surface Reflectivity [dB]', fontsize=16, color='g')
    ax1 = sns.lineplot(data=cygnss_df, x=date_column_name, y=sr_column_name, color='g')

    ax1.grid()

    ax2 = ax1.twinx()
    ax2.set_ylabel('Number of Samples', fontsize=16, color='b')
    ax2 = sns.barplot(x=list(cygnss_samples.keys()), y=list(cygnss_samples.values()), color='b', alpha=0.5)
    ax2.spines['right'].set_color('b')
    ax2.spines['left'].set_color('g')
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)

    try:
        plt.savefig('../Results/time_series/cygnss_samples.png')
    except:
        plt.savefig('Results/time_series/nice_time_series.png')
    plt.show()


def get_bar_plot_stats(cygnss_df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> dict:
    delta = datetime.timedelta(days=1)
    bar_plot_dict = {}
    while start_date <= end_date:
        bar_plot_dict[start_date.day] = len(cygnss_df[cygnss_df['dates'] == start_date.day])
        start_date += delta

    return bar_plot_dict


def main():
    kenya_soul_moisture_path = '../Data/ERA5/kenya_era5_sm.nc'
    india_rain_path = '/Data/ERA5/rain_india.nc'

    plot_era5_data(kenya_soul_moisture_path, pd.Timestamp(2020, 1, 1), target_value='swvl1', remove_lake=True)
    plot_era5_data(kenya_soul_moisture_path, pd.Timestamp(2020, 1, 1), target_value='swvl1', remove_lake=False)


if __name__ == '__main__':
    main()









