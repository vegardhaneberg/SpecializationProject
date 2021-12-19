import cartopy.crs as crs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches



##################################################
### Plotting Demonstration of CYGNSS SR values ###
##################################################
demo_df = pd.read_csv('/Volumes/DACOTA HDD/Semester Project CSV/Processed files/QF Removed/Processed2020-01-withoutQFs-5-25--15-45.csv')
demo_df = demo_df[demo_df['hours_after_jan_2020'] <= 3*24]
demo_df = scale_sr_values(demo_df)

fig = plt.figure(figsize=(9, 9))

# Settings for the plot
ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()
ax.add_feature(cfeature.LAKES, zorder=1, edgecolor='black')
ax.add_feature(cfeature.RIVERS, zorder=2)
sc = plt.scatter(demo_df['sp_lon'], demo_df['sp_lat'], c=list(demo_df['sr']), s=1, cmap='Spectral', zorder=3)
bar = plt.colorbar(sc)
bar.ax.set_title('SR [dB]')
ax.add_feature(cfeature.OCEAN, zorder=4)
ax.coastlines(zorder=5)

ax.set_extent([25, 45, 5, -15], crs=crs.PlateCarree()) ## Set overall area

lat_list, long_list = get_plot_ticks([5, -15], [25, 45])
ax.set_xticks(long_list, crs=ccrs.PlateCarree())
ax.set_yticks(lat_list, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

plt.title('Area of Interest for Demonstration', fontsize=18)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
# plt.savefig('/Users/madsrindal/Desktop/Plots/AreaOfInterestDemonstration.png')
plt.show()


#######################################################
### Plotting Demonstration Area of CYGNSS SR values ###
#######################################################
north = 50
west = -20
south = -50
east = 80

fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())

ax.stock_img()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
ax.set_extent([west, east, north, south], crs=crs.PlateCarree()) ## Set overall area

lat_list, long_list = get_plot_ticks([north, south], [west, east])
ax.set_xticks(long_list, crs=ccrs.PlateCarree())
ax.set_yticks(lat_list, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.add_patch(mpatches.Rectangle(xy=[25, 5], width=20, height=-20, facecolor='red', alpha=0.4, transform=ccrs.PlateCarree()))

plt.text(45.5, 0, '(45, 5)', horizontalalignment='left', transform=ccrs.Geodetic(), fontsize=9, color='black', weight='bold')
plt.text(24, -15, '(25, -15)', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=9, color='black', weight='bold')

plt.title('Area of Interest for Demonstration', fontsize=18)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

plt.plot(25, 5, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(25, -15, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(45, 5, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(45, -15, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
#plt.savefig('/Users/madsrindal/Desktop/Plots/AreaOfInterestAfrica.png')
plt.show()


#####################################################################
### Plotting Area of Interest for ML and Incidence Angle Analysis ###
#####################################################################
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


fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())

ax.stock_img()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
ax.set_extent([50, 70, 20, 40], crs=crs.PlateCarree()) ## Set overall area

lat_list, long_list = get_plot_ticks([20, 40], [50, 70])
ax.set_xticks(long_list, crs=ccrs.PlateCarree())
ax.set_yticks(lat_list, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.add_patch(mpatches.Rectangle(xy=[61, 25], width=2, height=2, facecolor='red', alpha=0.4, transform=ccrs.PlateCarree()))

plt.text(57, 32, 'IRAN', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20)
plt.text(69.5, 27.5, 'PAKISTAN', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=18)

plt.text(60.7, 27, '(61, 27)', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=13, color='black', weight='bold')
plt.text(63.5, 24.5, '(63, 25)', horizontalalignment='left', transform=ccrs.Geodetic(), fontsize=13, color='black', weight='bold')

plt.title('Chosen Area of Interest', fontsize=18)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

plt.plot(61, 25, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(61, 27, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(63, 25, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(63, 27, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
# plt.savefig('/Users/madsrindal/Desktop/Plots/AreaOfInterestIranPakistan.png')
plt.show()



##########################################################################
### Plotting Area of Interest for Correlation Analysis and Time Series ###
##########################################################################
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


fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())

ax.stock_img()
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle='-', alpha=.5)
ax.set_extent([60, 90, 15, 45], crs=crs.PlateCarree()) ## Set overall area

lat_list, long_list = get_plot_ticks([15, 45], [60, 90])
ax.set_xticks(long_list, crs=ccrs.PlateCarree())
ax.set_yticks(lat_list, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.add_patch(mpatches.Rectangle(xy=[69.9, 24.7], width=9.9, height=7.8, facecolor='red', alpha=0.4, transform=ccrs.PlateCarree()))

plt.text(81, 20, 'INDIA', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=20)
plt.text(69.5, 27.5, 'PAKISTAN', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=16)

plt.text(69.5, 32, '(69.9, 32.5)', horizontalalignment='right', transform=ccrs.Geodetic(), fontsize=13, color='black', weight='bold')
plt.text(80, 23.5, '(79.8, 24.7)', horizontalalignment='left', transform=ccrs.Geodetic(), fontsize=13, color='black', weight='bold')

plt.title('Chosen Area of Interest', fontsize=18)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

plt.plot(69.9, 32.5, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(69.9, 24.7, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(79.8, 32.5, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
plt.plot(79.8, 24.7, color='red', marker='o', alpha=0.4, transform=ccrs.PlateCarree())
# plt.savefig('/Users/madsrindal/Desktop/Plots/AreaOfInterestPakistanIndia.png')
plt.show()


##################################################
### Plotting Incidence Angle Distribution 2021 ###
##################################################
print('Min: ', interpolated_df_2021['sp_inc_angle'].min())
print('Max: ', interpolated_df_2021['sp_inc_angle'].max())

plt.hist(interpolated_df_2021['sp_inc_angle'])
plt.title('Incidence Angle Measurements 2020', fontsize=18)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Incidence angle', fontsize=12)
# plt.savefig('/Users/madsrindal/Desktop/Plots/IncidenceAngleDistribution2020.png', bbox_inches='tight')
plt.show()


#####################################################
### Plotting Correlation Based on Incidence Angle ###
#####################################################
corr_list = []
inc_angle = []

for i in range(0, 71, 2):
    chosen_df = interpolated_df_2020[interpolated_df_2020['sp_inc_angle'] >= i]
    chosen_df = chosen_df[chosen_df['sp_inc_angle'] <= i+20]
    corr = chosen_df['smap_sm'].corr(chosen_df['sr'])
    corr_list.append(corr)
    inc_angle.append(i)

print(max(corr_list))
plt.plot(inc_angle, corr_list, linewidth=4.0)
plt.title('SMAP SM and SR Correlation 2021', fontsize=18)
plt.ylabel('Correlation', fontsize=12)
plt.xlabel('Incidence angle', fontsize=12)
# plt.savefig('/Users/madsrindal/Desktop/Plots/IncidenceAngleCorrelation2020smap.png')
plt.show()


#########################################################
### Plotting ERA5 Incidence Angle [0-20] Scatter Plot ###
#########################################################
chosen_df = interpolated_df_2020[interpolated_df_2020['sp_inc_angle'] >= 0]
chosen_df = chosen_df[chosen_df['sp_inc_angle'] <= 20]
sc = plt.scatter(chosen_df['sm'], chosen_df['sr'], c=chosen_df['sp_inc_angle'], s=10)
bar = plt.colorbar(sc)
bar.ax.set_title('Incidence Angle')
plt.ylabel('Surface Reflectivity [dB]', fontsize=12)
plt.xlabel('ERA5 Soil Moisture [m^3/m^3]', fontsize=12)
# plt.savefig('/Users/madsrindal/Desktop/Plots/SRandSM_era5_inc_angle_0-20_2020.png')
plt.show()


############################################################
### Plotting SMAP RMSE Based On Incidence Angle Interval ###
############################################################
model_list = ['Linear Regression', 'Huber Regression', 'RANSAC Regression', 'Theil-Sen Regression', 'CatBoost', 'XGBoost', 'GBM', 'DRF']

for model in model_list:
    x = list(rmse_table_smap.columns)
    x.remove('Model')
    y = rmse_table_smap[rmse_table_smap['Model'] == model].values.tolist()[0]
    y.remove(model)
    
    plt.plot(x, y, linewidth=3, label=model)
    plt.title('RMSE of Tested SMAP Prediction Models', fontsize=18)
    plt.ylabel('RMSE [cm^3/cm^3]', fontsize=12)
    plt.xlabel('Incidence Angle Interval', fontsize=12)

plt.legend(fontsize='small')
# plt.savefig('/Users/madsrindal/Desktop/Plots/ml_models_rmse.png')
plt.show()


############################################################
### Plotting ERA RMSE Based On Incidence Angle Interval ###
############################################################
model_list = ['Linear Regression', 'Huber Regression', 'RANSAC Regression', 'Theil-Sen Regression']

for model in model_list:
    x = list(rmse_table_era5.columns)
    x.remove('Model')
    y = rmse_table_era5[rmse_table_era5['Model'] == model].values.tolist()[0]
    y.remove(model)
    
    plt.plot(x, y, linewidth=4, label=model)
    plt.title('RMSE of Tested ERA5 Prediction Models', fontsize=18)
    plt.ylabel('RMSE [m^3/m^3]', fontsize=12)
    plt.xlabel('Incidence Angle Interval', fontsize=12)

plt.legend(fontsize='small')
# plt.savefig('/Users/madsrindal/Desktop/Plots/ml_models_rmse_era5.png')
plt.show()


#######################################
### Plotting SMAP Density Histogram ###
#######################################
hist = plt.hist2d(interpolated_df_2020['smap_sm'], interpolated_df_2020['sr'], (20, 20), cmap=plt.cm.jet, norm=LogNorm(), cmin=1)
plt.title('SMAP SM and SR Histogram', fontsize=18)
plt.ylabel('Surface Reflectivity [dB]', fontsize=12)
plt.xlabel('SMAP Soil Moisture [cm^3/cm^3]', fontsize=12)
bar = plt.colorbar()
bar.ax.set_title('Density')
# plt.savefig('/Users/madsrindal/Desktop/Plots/smap_sm_and_sr_density_histogram.png')
plt.show()


#######################################################
### Plotting SMAP-ERA5 Comparison Density Histogram ###
#######################################################
hist = plt.hist2d(interpolated_df_2020['smap_sm'], interpolated_df_2020['sm'], (70, 70), cmap=plt.cm.jet, norm=LogNorm(), cmin=1)
plt.title('SM Density Histogram', fontsize=18)
plt.xlabel('SMAP Soil Moisture [cm^3/cm^3]', fontsize=12)
plt.ylabel('ERA5 Soil Moisture [m^3/m^3]', fontsize=12)
bar = plt.colorbar()
bar.ax.set_title('Density')
# plt.savefig('/Users/madsrindal/Desktop/Plots/era5_and_smap_sm_density_histogram.png')
plt.show()