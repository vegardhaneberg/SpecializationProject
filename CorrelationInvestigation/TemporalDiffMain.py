import datetime

import pandas as pd
try:
    from cygnss.cygnss import filter_location
except:
    from cygnss import filter_location
import numpy as np
import time
from matplotlib import pyplot as plt

from plot import universal_plot, plot_era5_data
from era5 import read_era5_data, filter_era5_location, convert_date_to_hours
from scipy.ndimage import gaussian_filter
from plotly import express as px
import math
from random import random
from matplotlib.colors import LogNorm


def plot_corr(x, y, diff=False, month='Difference'):

    m, b = np.polyfit(x, y, 1)

    fig, ax = plt.subplots()
    if diff:
        fig.suptitle("CYGNSS and ERA5 Histogram " + month, fontsize=18)
    else:
        fig.suptitle("CYGNSS and ERA5 Histogram " + month + " 2020", fontsize=18)
    hist = ax.hist2d(x, y, (70, 70), cmap=plt.cm.jet, norm=LogNorm(), cmin=1)
    line, = plt.plot(x, m * x + b, 'red')
    line.set_label('Linear Fit: ' + str(round(m, 2)) + 'X + ' + str(round(b, 2)))
    line.set_linewidth(3)
    ax.legend()

    bar = fig.colorbar(hist[3], ax=ax)
    bar.ax.set_title("Count in log scale")
    if diff:
        ax.set_xlabel("ERA5 Difference SM", fontsize=14)
        ax.set_ylabel("CYGNSS Difference SR", fontsize=14)
    else:
        ax.set_xlabel("ERA5 SM", fontsize=14)
        ax.set_ylabel("CYGNSS SR", fontsize=14)

    save_filename = str(math.floor(random()*100000)) + '.png'
    print(save_filename)
    plt.savefig(save_filename)
    plt.show()


def read_and_gridbox_cygnss_data(path, area, from_date=None, to_date=None, smooth=None, angle_cut_of=None):
    df = pd.read_csv(path)

    df = df[['sp_lat', 'sp_lon', 'sp_inc_angle', 'sr']]
    df = filter_location(df, [area['north'], area['west'], area['south'], area['east']])

    if angle_cut_of is not None:
        df = df[df['sp_inc_angle'] <= angle_cut_of]

    df['sp_lat'] = df['sp_lat'].apply(lambda x: round(x, 1))
    df['sp_lon'] = df['sp_lon'].apply(lambda x: round(x, 1))

    if to_date is not None and from_date is not None:
        start_hour = convert_date_to_hours(datetime.date(2020, 1, 1), from_date)
        end_hour = convert_date_to_hours(datetime.date(2020, 1, 1), to_date)
        df = df[df['hours_after_jan_2020'] >= start_hour]
        df = df[df['hours_after_jan_2020'] <= end_hour]

    avg_df = pd.DataFrame(columns=['lat', 'long', 'avg_sr', 'samples', 'std'])
    missing_value_coords = []

    print('-----------------------------------------------------')
    print('Starting to calculate average values CYGNSS...')
    for lat in np.arange(area['north'], area['south'] - 0.05, -0.1):
        for long in np.arange(area['west'], area['east'] + 0.05, 0.1):
            tmp_df = df[df['sp_lat'] == round(lat, 1)]
            tmp_df = tmp_df[tmp_df['sp_lon'] == round(long, 1)]
            samples = len(tmp_df)
            if samples == 0:
                print('Coordinate without cygnss samples:', round(lat, 1), round(long, 1))
                missing_value_coords.append((round(lat, 1), round(long, 1)))
                # avg_df = avg_df.append(dict(zip(avg_df.columns, [round(lat, 1), round(long, 1), 0, 0])), ignore_index=True)
            else:
                avg_sr = sum(list(tmp_df['sr']))/samples
                tmp_std = np.std(list(tmp_df['sr']))
                avg_df = avg_df.append(dict(zip(avg_df.columns, [round(lat, 1), round(long, 1), avg_sr, samples, tmp_std])), ignore_index=True)

    # Converting to dB
    min_sr = avg_df['avg_sr'].min()
    avg_df['avg_sr'] = avg_df['avg_sr'].apply(lambda x: x - min_sr)

    fill_value = np.average(avg_df['avg_sr'])

    for missing_coord in missing_value_coords:
        avg_df = avg_df.append(dict(zip(avg_df.columns, [missing_coord[0], missing_coord[1], fill_value, 0, 0])), ignore_index=True)
    print('Finished averaging CYGNSS data')
    print('Length of dataframe:', len(avg_df))
    print('Avg samples in each grid box:', np.mean(list(avg_df['samples'])))
    print('Avg SR', np.mean(list(avg_df['avg_sr'])))
    print('Std between cells:', np.std(list(avg_df['avg_sr'])))
    print('-----------------------------------------------------')

    if smooth is not None:
        avg_df = smoothening(avg_df, area, sigma=smooth, target_value='avg_sr')

    return avg_df, missing_value_coords


def smoothening(df: pd.DataFrame, area: dict, sigma: float, target_value='swvl1') -> pd.DataFrame:
    df = df.sort_values(['lat', 'long'], ascending=(False, True))
    lats = np.arange(area['north'], area['south'] - 0.05, -0.1)
    lats = np.around(lats, 1)
    longs = np.arange(area['west'], area['east'] + 0.05, 0.1)
    longs = np.around(longs, 1)

    target_values = np.array(df[target_value]).reshape(len(lats), len(longs))
    target_values = gaussian_filter(target_values, sigma=sigma)
    target_values = target_values.flatten()
    df[target_value] = target_values

    return df


def read_and_average_era5_data(path, area, start_date, end_date, target_value='swvl1'):
    print('Reading ERA5 data...')
    df = read_era5_data(path, target_value, remove_lake=True)
    print('Done reading ERA5 data')

    # Filtering on location
    df = filter_era5_location(df, area, include_equals=True)

    # Filtering on time
    reference_date = datetime.date(2020, 1, 1)
    start_hour = convert_date_to_hours(reference_date, start_date)
    end_hour = convert_date_to_hours(reference_date, end_date)
    df = df[df['time'] >= start_hour]
    df = df[df['time'] <= end_hour]

    print('Starting to calculate average values ERA5...')
    # Calculating average values
    avg_df = pd.DataFrame(columns=['lat', 'long', 'avg_sm', 'samples'])
    lake_coordinates = []
    for lat in np.arange(area['north'], area['south'] - 0.05, -0.1):
        for long in np.arange(area['west'], area['east'] + 0.05, 0.1):
            tmp_df = df[df['lat'] == round(lat, 1)]
            tmp_df = tmp_df[tmp_df['long'] == round(long, 1)]
            samples = len(tmp_df)
            if samples == 0:
                print('ERA5 coordinate without samples:', round(lat, 1), round(long, 1))
                lake_coordinates.append((round(lat, 1), round(long, 1)))
                # avg_df = avg_df.append(dict(zip(avg_df.columns, [round(lat, 1), round(long, 1), 0, 0])), ignore_index=True)
            else:
                avg_sm = sum(list(tmp_df[target_value]))/samples
                avg_df = avg_df.append(dict(zip(avg_df.columns, [round(lat, 1), round(long, 1), avg_sm, samples])), ignore_index=True)
    print('Finished averaging ERA5')
    print('Length of dataframe:', len(avg_df))
    print('Avg samples in each grid box:',  round(np.mean(list(avg_df['samples'])), 2))
    print('Avg SM', np.mean(list(avg_df['avg_sm'])))
    print('Std between cells:', round(np.std(list(avg_df['avg_sm'])), 2))

    return avg_df, lake_coordinates


def main():
    s = time.time()
    era5_target = 'swvl1'
    use_smoothening = 1
    angle = 60

    test_area = {'north': 32.5, 'south': 24.7, 'west': 69.6, 'east': 79.8}

    # January
    cygnss_jan, missing_cygnss_jan = read_and_gridbox_cygnss_data('/Users/vegardhaneberg/Desktop/Jan2020-38-45-8-85.csv',
                                                                  test_area,
                                                                  smooth=use_smoothening,
                                                                  angle_cut_of=angle)
    era5_jan, lake_coords_jan = read_and_average_era5_data('../Data/ERA5/era5_rain_sm_india_january_2020', test_area,
                                                           datetime.date(2020, 1, 1),
                                                           datetime.date(2020, 1, 31),
                                                           target_value=era5_target)


    # August
    cygnss_aug, missing_cygnss_aug = read_and_gridbox_cygnss_data('/Users/vegardhaneberg/Desktop/Aug2020-38-45-8-85.csv',
                                                                  test_area,
                                                                  smooth=use_smoothening,
                                                                  angle_cut_of=angle)
    era5_aug, lake_coords_aug = read_and_average_era5_data('../Data/ERA5/era5_rain_sm_india_august_2020', test_area,
                                                           datetime.date(2020, 8, 1),
                                                           datetime.date(2020, 8, 31),
                                                           target_value=era5_target)
    print('-----------------------------------------------------')

    vmin_cygnss = min(min(list(cygnss_jan['avg_sr'])), min(list(cygnss_aug['avg_sr'])))
    vmax_cygnss = max(max(list(cygnss_jan['avg_sr'])), max(list(cygnss_aug['avg_sr'])))

    vmin_era5 = min(min(list(era5_jan['avg_sm'])), min(list(era5_aug['avg_sm'])))
    vmax_era5 = max(max(list(era5_jan['avg_sm'])), max(list(era5_aug['avg_sm'])))

    # January Plot
    universal_plot(era5_jan, 'avg_sm', title='ERA5 SM January 2020', bar_title='Soil Moisture m^3/m^3', save=True,
                   vmin=vmin_era5, vmax=vmax_era5)
    universal_plot(cygnss_jan, 'avg_sr', title="CYGNSS SR January 2020", bar_title="Surface Reflectivity [dB]", save=True,
                   vmin=vmin_cygnss, vmax=vmax_cygnss)
    universal_plot(cygnss_jan, 'std', title=" CYGNSS STD January 2020", bar_title="STD [dB]", std=True, dot_size=3)

    # August Plot
    universal_plot(era5_aug, 'avg_sm', title='ERA5 SM August 2020', bar_title='Soil Moisture m^3/m^3', save=True,
                   vmin=vmin_era5, vmax=vmax_era5)
    universal_plot(cygnss_aug, 'avg_sr', title='CYGNSS SR August 2020', bar_title='Surface Reflectivity [dB]', save=True,
                   vmin=vmin_cygnss, vmax=vmax_cygnss)
    universal_plot(cygnss_aug, 'std', title=" CYGNSS STD August 2020", bar_title="STD [dB]", std=True, dot_size=3)

    # January Correlation
    merged_jan = pd.merge(era5_jan, cygnss_jan, on=['lat', 'long'], how='inner')
    corr_jan = merged_jan['avg_sm'].astype('float64').corr(merged_jan['avg_sr'].astype('float64'))
    cygnss_jan_avg = cygnss_jan['avg_sr'].mean()
    era5_jan_avg = era5_jan['avg_sm'].mean()

    # August Correlation
    merged_aug = pd.merge(era5_aug, cygnss_aug, on=['lat', 'long'], how='inner')
    corr_aug = merged_aug['avg_sm'].astype('float64').corr(merged_aug['avg_sr'].astype('float64'))
    cygnss_aug_avg = cygnss_aug['avg_sr'].mean()
    era5_aug_avg = era5_aug['avg_sm'].mean()

    # Temporal Difference
    cygnss_jan = cygnss_jan.rename(columns={'avg_sr': 'avg_sr_jan'})
    cygnss_aug = cygnss_aug.rename(columns={'avg_sr': 'avg_sr_aug'})
    era5_jan = era5_jan.rename(columns={'avg_sm': 'era5_sm_jan'})
    era5_aug = era5_aug.rename(columns={'avg_sm': 'era5_sm_aug'})

    merged_cygnss = pd.merge(cygnss_jan, cygnss_aug, on=['lat', 'long'], how='inner')
    merged_era5 = pd.merge(era5_jan, era5_aug, on=['lat', 'long'], how='inner')
    merged_cygnss['diff_cygnss'] = merged_cygnss['avg_sr_aug'] - merged_cygnss['avg_sr_jan']
    merged_era5['diff_era5'] = merged_era5['era5_sm_aug'] - merged_era5['era5_sm_jan']

    difference_merged = pd.merge(merged_cygnss, merged_era5, on=['lat', 'long'], how='inner')

    # Temporal Difference Correlation
    corr_diff = difference_merged['diff_cygnss'].astype('float64').corr(difference_merged['diff_era5'].astype('float64'))

    print('Correlaton January:', corr_jan)
    print('Correlaton August:', corr_aug)
    print('Correlaton Difference:', corr_diff)

    universal_plot(difference_merged, 'diff_cygnss', title='CYGNSS Difference SR Jan-Aug 2020', bar_title='Surface Reflectivity [dB]', save=True)
    universal_plot(difference_merged, 'diff_era5', title='ERA5 Difference SM Jan-Aug 2020', bar_title='Soil Moisture m^3/m^3', save=True)

    plot_corr(merged_jan['avg_sm'], merged_jan['avg_sr'], diff=False, month='January')
    plot_corr(merged_aug['avg_sm'], merged_aug['avg_sr'], diff=False, month='August')
    plot_corr(difference_merged['diff_era5'], difference_merged['diff_cygnss'], diff=True, month='Difference')

    print('AVG SR Jan CYGNSS:', cygnss_jan_avg)
    print('AVG SR Aug CYGNSS:', cygnss_aug_avg)
    print('Difference CYGNSS:', cygnss_jan_avg - cygnss_aug_avg)
    print('AVG SM Jan ERA5:', era5_jan_avg)
    print('AVG SM Aug ERA5:', era5_aug_avg)
    print('Difference ERA5:', era5_jan_avg - era5_aug_avg)








    """
    missing_coords = []
    missing_coords.extend(missing_cygnss_jan)
    missing_coords.extend(missing_cygnss_aug)
    missing_coords.extend(lake_coords_jan)
    missing_coords.extend(lake_coords_aug)

    for coord in missing_coords:
        cygnss_jan = cygnss_jan[(cygnss_jan['lat'] != coord[0]) | (cygnss_jan['long'] != coord[1])]
        cygnss_aug = cygnss_aug[(cygnss_aug['lat'] != coord[0]) | (cygnss_aug['long'] != coord[1])]

        era5_jan = era5_jan[(era5_jan['lat'] != coord[0]) | (era5_jan['long'] != coord[1])]
        era5_aug = era5_aug[(era5_aug['lat'] != coord[0]) | (era5_aug['long'] != coord[1])]

    if not (len(cygnss_jan) == len(cygnss_aug) == len(era5_jan) == len(era5_aug)):
        print('###############################')
        print('WARNING! MISMATCH ON DF LENGTHS')
        print('###############################')

    vmin_cygnss_jan = min(list(cygnss_jan['avg_sr']))
    vmin_cygnss_aug = min(list(cygnss_aug['avg_sr']))
    vmin_cygnss = min(vmin_cygnss_jan, vmin_cygnss_aug)

    vmax_cygnss_jan = max(list(cygnss_jan['avg_sr']))
    vmax_cygnss_aug = max(list(cygnss_aug['avg_sr']))
    vmax_cygnss = max(vmax_cygnss_jan, vmax_cygnss_aug)

    vmin_era5_jan = min(list(era5_jan['avg_sm']))
    vmin_era5_aug = min(list(era5_aug['avg_sm']))
    vmin_era5 = min(vmin_era5_jan, vmin_era5_aug)

    vmax_era5_jan = max(list(era5_jan['avg_sm']))
    vmax_era5_aug = max(list(era5_aug['avg_sm']))
    vmax_era5 = max(vmax_era5_jan, vmax_era5_aug)

    universal_plot(cygnss_jan, 'avg_sr', title="CYGNSS SR January", bar_title="Surface Reflectivity [dB]",
                   vmin=vmin_cygnss, vmax=vmax_cygnss)
    universal_plot(cygnss_jan, 'std', title=" CYGNSS STD January", bar_title="STD [dB]")
    universal_plot(cygnss_aug, 'avg_sr', title="CYGNSS SR August", bar_title="Surface Reflectivity [dB]",
                   vmin=vmin_cygnss, vmax=vmax_cygnss)
    universal_plot(cygnss_aug, 'std', title="CYGNSS STD August", bar_title="STD [dB]")
    universal_plot(era5_jan, 'avg_sm', title="ERA5 SM January", bar_title="Soil Moisture [m^3/m^3]",
                   vmin=vmin_era5, vmax=vmax_era5)
    universal_plot(era5_aug, 'avg_sm', title="ERA5 SM August", bar_title="Soil Moisture [m^3/m^3]",
                   vmin=vmin_era5, vmax=vmax_era5)

    cygnss_jan = cygnss_jan.sort_values(['lat', 'long'], ascending=(False, True))
    cygnss_aug = cygnss_aug.sort_values(['lat', 'long'], ascending=(False, True))
    era5_jan = era5_jan.sort_values(['lat', 'long'], ascending=(False, True))
    era5_aug = era5_aug.sort_values(['lat', 'long'], ascending=(False, True))

    jan_corr = cygnss_jan['avg_sr'].corr(era5_jan['avg_sm'])
    aug_corr = cygnss_aug['avg_sr'].corr(era5_aug['avg_sm'])

    cygnss_jan = cygnss_jan.rename(columns={'avg_sr': 'avg_sr_jan'})
    cygnss_aug = cygnss_aug.rename(columns={'avg_sr': 'avg_sr_aug'})
    era5_jan = era5_jan.rename(columns={'avg_sm': 'avg_sm_jan'})
    era5_aug = era5_aug.rename(columns={'avg_sm': 'avg_sm_aug'})

    cygnss_merged = pd.merge(cygnss_jan, cygnss_aug, on=['lat', 'long'], how='inner')
    era5_merged = pd.merge(era5_jan, era5_aug, on=['lat', 'long'], how='inner')

    master_merged = pd.merge(cygnss_merged, era5_merged, on=['lat', 'long'], how='inner')

    master_merged['cygnss_diff'] = master_merged.apply(lambda row: row['avg_sr_jan'] - row['avg_sr_aug'], axis=1)
    master_merged['era5_diff'] = master_merged.apply(lambda row: row['avg_sm_jan'] - row['avg_sm_aug'], axis=1)

    universal_plot(master_merged, target_value='cygnss_diff', title='MERGED CYGNSS DELTA SR', bar_title="Delta SR [dB]")
    universal_plot(master_merged, target_value='era5_diff', title="ERA5 Delta SM", bar_title="Delta SM[m^3/m^3]")

    diff_corr = master_merged['cygnss_diff'].corr(master_merged['era5_diff'])

    print()
    print('Correlation January:', round(jan_corr, 3))
    print('Correlation August:', round(aug_corr, 3))
    print('Correlation Difference:', round(diff_corr, 3))

    plot_corr(era5_jan['avg_sm_jan'], cygnss_jan['avg_sr_jan'], diff=False, month='January')
    plot_corr(era5_aug['avg_sm_aug'], cygnss_aug['avg_sr_aug'], diff=False, month='August')
    plot_corr(master_merged['era5_diff'], master_merged['cygnss_diff'], diff=True, month='Difference')

    fig = px.density_heatmap(master_merged, x='cygnss_diff', y='era5_diff', nbinsx=50, nbinsy=50,
                             marginal_x="histogram", marginal_y="histogram")
    fig.show()
    print('Total time:', round(time.time() - s, 1), 'seconds')
    """

if __name__ == '__main__':
    main()








