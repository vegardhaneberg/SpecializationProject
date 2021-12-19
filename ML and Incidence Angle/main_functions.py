# Set boolean options
collect_data_boolean = False
process_cygnss_data_boolean = False
generate_interpolation_function_boolean = False
interpolate_dataframe_boolean = False

#############################################
### SET INTERPOLATION FUNCTION PARAMETERS ###
#############################################

location = [27, 61, 25, 63]
year = 2020
location_string = str(location[0]) + '-' + str(location[1]) + '-' + str(location[2]) + '-' + str(location[3])
era5_df = read_era5_data('/Volumes/DACOTA HDD/Semester Project CSV/ERA5/era5_rain_sm_' + str(year) + '_' + location_string + '.nc', year=year)
cygnss_storage_folder = '/Volumes/DACOTA HDD/Semester Project CSV/Processed files/Containing QF/' + location_string
interpolation_function_storage_folder = '/Volumes/DACOTA HDD/Semester Project CSV/Interpolation functions'
interpolated_df_storage_folder = '/Volumes/DACOTA HDD/Semester Project CSV/Processed files/Interpolated dfs/' + location_string

# IF COLLECTING FOR MORE THAN ONE MONTH
start_month = 1
end_month = 12

# IF COLLECTING FOR ONE MONTH ONLY
single_month_boolean = False
single_month = 12



#############################################################
#### COLLECT DATA FOR A NUMBER OF DAYS IN A SINGLE MONTH ####
#############################################################

if collect_data_boolean:
    year = 2020
    month = 7
    num_of_days = monthrange(year, month)[1]
    start_day = 1
    end_day = num_of_days

    days_with_error = []

    raw_main_df = pd.DataFrame()

    for i in range(start_day, end_day+1):  # Number of days to collect data
        print('#############################################################')
        print('#############################################################')
        print('Starting computation for day ' + str(i) + ' of '+ str(end_day) + '...........')
        raw_df, raw_data_list, failed_satellites = create_cygnss_df(2020, month, i)

        if len(failed_satellites) > 0:
            days_with_error.append([i, failed_satellites])

        print('-------------------------------------------------------------')

        print('Removing fill values...')
        rows_before_removal = raw_df.shape[0]
        satellite_df = raw_df

        for j in range(len(raw_data_list)):  # Number of satellites
            satellite_df = remove_fill_values(raw_df, raw_data_list[j])

        rows_after_removal = satellite_df.shape[0]
        print('Removed ' + str(rows_before_removal - rows_after_removal) + ' rows containing fill values...')

        satellite_df.to_csv("/Users/madsrindal/Downloads/raw_main_df_2020_" + str(month).zfill(2) + "_" + str(i) + "of" + str(end_day) + ".csv", index=False)

        print('Day: ', i)
        print('Failed satellites: ', failed_satellites)
        print('Days with error in total: ', days_with_error)

        print('#############################################################')
        print('#############################################################\n\n')


###########################################################################
### PROCESS DOWNLOADED CYGNSS DATA FOR A SINGLE MONTH OR AN ENTIRE YEAR ###
###########################################################################

if process_cygnss_data_boolean:
    
    # SET PARAMETERS
    location = [5, 25, -15, 45]
    location_string = str(location[0]) + '-' + str(location[1]) + '-' + str(location[2]) + '-' + str(location[3])
    year = 2020
    month = 1
    retrieval_folder = '/Volumes/DACOTA HDD/Semester Project CSV/CYGNSS ' + str(year) + '-'
    storage_path = '/Volumes/DACOTA HDD/Semester Project CSV/Processed files/QF Removed'
    features = []
    single_month = True

    if single_month:
        print('*' * 39)
        print('*' * 14 + f'{month_name[month]:^11}' + '*' * 14)
        print('*' * 39 + '\n')
        monthly_df = process_monthly_df(retrieval_folder, year, month, location, True, False)
        filtered_monthly_df = select_df_features(monthly_df, features)
        full_storage_path = storage_path + '/Processed' + str(year) + '-' + str(month).zfill(2) + '-withQFs-' + \
                            location_string + '.csv'
        store_df_as_csv(filtered_monthly_df, full_storage_path)
        
    else:
        for i in range(12):
            print('*' * 39)
            print('*' * 14 + f'{month_name[i + 1]:^11}' + '*' * 14)
            print('*' * 39 + '\n')
            monthly_df = process_monthly_df(retrieval_folder, year, i + 1, location, True, False)
            filtered_monthly_df = select_df_features(monthly_df, features)
            full_storage_path = storage_path + '/Processed' + str(year) + '-' + str(month).zfill(2) + '-withQFs-' + \
                            location_string + '.csv'
            store_df_as_csv(filtered_monthly_df, full_storage_path)


########################################
### GENERATE INTERPOLATION FUNCTIONS ###
########################################

if generate_interpolation_function_boolean:
    if single_month_boolean:
        monthly_interpolation_function = generate_monthly_interpolation_function(era5_df, location, single_month, year, interpolation_function_storage_folder)
        print('------------------------------------------------------------------------')
        print('Saving the interpolation function as a pickle file...')

        with open(interpolation_function_storage_folder + '/interpolation_function_' + str(year) + '_' + str(single_month).zfill(2) + '_' + location_string + '.pickle', 'wb') as pickle_interpolation_file:
            pickle.dump(monthly_interpolation_function, pickle_interpolation_file)
        print('Saved interpolation file as pickle file on drive...\n\n')
        
    else:
        for i in range(start_month, end_month + 1):
            print('*' * 39)
            print('*' * 14 + f'{month_name[i]:^11}' + '*' * 14)
            print('*' * 39)
            monthly_interpolation_function = generate_monthly_interpolation_function(era5_df, location, i, year, interpolation_function_storage_folder)
            print('------------------------------------------------------------------------')
            print('Saving the interpolation function as a pickle file...')

            with open(interpolation_function_storage_folder + '/interpolation_function_' + str(year) + '_' + str(i).zfill(2) + '_' + location_string + '.pickle', 'wb') as pickle_interpolation_file:
                pickle.dump(monthly_interpolation_function, pickle_interpolation_file)
            print('Saved interpolation file as pickle file on drive...\n\n')

else:
    if single_month_boolean:
        monthly_interpolation_function = load_monthly_interpolation_function(location, single_month, year, interpolation_function_storage_folder)
    else:
        interpolation_functions_2020 = []
        interpolation_functions_2021 = []
        for i in range(start_month, end_month + 1):
            monthly_interpolation_function_2020 = load_monthly_interpolation_function(location, i, year, interpolation_function_storage_folder)
            interpolation_functions_2020.append(monthly_interpolation_function_2020)
        for i in range(start_month, end_month-9):
            monthly_interpolation_function_2021 = load_monthly_interpolation_function(location, i, year+1, interpolation_function_storage_folder)
            interpolation_functions_2021.append(monthly_interpolation_function_2021)


rootdir = '/Users/madsrindal/Desktop/5000002644104'
df_2020 = get_smap_df_year(rootdir, 2020, convert_time_hours=True)
df_2021 = get_smap_df_year(rootdir, 2021, convert_time_hours=True)
func_2020 = interpolate_ml(df_2020, target_value='smap_sm')
func_2021 = interpolate_ml(df_2021, target_value='smap_sm')


#########################################
### GENERATE INTERPOLATED DATA FRAMES ###
#########################################

# Interpolate ERA5 data
if interpolate_dataframe_boolean:
    if single_month_boolean:
        interpolated_df = generate_monthly_interpolated_cygnss_df(location, single_month, year, cygnss_storage_folder, 
                                                                  monthly_interpolation_function)
        print('------------------------------------------------------------------------')
        print('Saving the interpolated dataframe as a csv file')
        if year != 2020:
            interpolated_df.to_csv(interpolated_df_storage_folder + '/df_with_interpolated_sm_' + str(year) + '_' + str(single_month).zfill(2) + '_' + location_string + '.csv', index=False)
        else:
            interpolated_df.to_csv(interpolated_df_storage_folder + '/df_with_interpolated_sm_' + str(single_month).zfill(2) + '_' + location_string + '.csv', index=False)
    
    else:
        # cygnss_df = pd.DataFrame()
        for i in range(start_month, end_month + 1):
            monthly_df = generate_monthly_interpolated_cygnss_df(location, i, year, cygnss_storage_folder, 
                                                interpolation_functions[i-start_month])
            print('------------------------------------------------------------------------')
            print('Saving the interpolated dataframe as a csv file\n\n')
            if year != 2020:
                monthly_df.to_csv(interpolated_df_storage_folder + '/df_with_interpolated_sm_' + str(year) + '_' + str(i).zfill(2) + '_' + location_string + '.csv', index=False)
            else:
                monthly_df.to_csv(interpolated_df_storage_folder + '/df_with_interpolated_sm_' + str(i).zfill(2) + '_' + location_string + '.csv', index=False)
else:
    if single_month_boolean:
        interpolated_df = load_monthly_interpolated_df(location, single_month, year, interpolated_df_storage_folder)
    else:
        interpolated_df_2020 = load_periodic_interpolated_df(location, start_month, end_month, year, interpolated_df_storage_folder)
        interpolated_df_2021 = load_periodic_interpolated_df(location, start_month, end_month-10, year+1, interpolated_df_storage_folder)

# Interpolate SMAP data
def interpolate_smap_df(df, interpolation_function):
    df['smap_sm'] = df.apply(lambda row: interpolation_function(row.hours_after_jan_2020, row.sp_lat, row.sp_lon), axis=1)
    return df

interpolated_df_2021['hours_after_jan_2020'] = interpolated_df_2021['hours_after_jan_2020']+(366*24)
interpolated_df_2020 = interpolate_smap_df(interpolated_df_2020, func_2020)
interpolated_df_2021 = interpolate_smap_df(interpolated_df_2021, func_2021)


###########################
### PROCESS DATA FRAMES ###
###########################

## FILTER QUALITY FLAGS
interpolated_df_2020 = filter_quality_flags_1(interpolated_df_2020)
interpolated_df_2021 = filter_quality_flags_1(interpolated_df_2021)

## COMPUTE BLOCK_CODE
interpolated_df_2020 = compute_block_code(interpolated_df_2020)
interpolated_df_2021 = compute_block_code(interpolated_df_2021)

## SCALE SURFACE REFLECTIVITY VALUES (REMOVE THE MIN VALUE FROM ALL OTHER VALUES)
interpolated_df_2020 = scale_sr_values(interpolated_df_2020)
interpolated_df_2021 = scale_sr_values(interpolated_df_2021)

## COMPUTE NEW UNIQUE TRACK IDS (IF NECESSARY)
interpolated_df_2020 = compute_unique_track_ids(interpolated_df_2020)

## FILTER ERA5 NAN VALUES
interpolated_df_2020 = filter_nan_era5(interpolated_df_2020)
interpolated_df_2021 = filter_nan_era5(interpolated_df_2021)

## FILTER SMAP NAN VALUES
interpolated_df_2020 = filter_nan_smap(interpolated_df_2020)
interpolated_df_2021 = filter_nan_smap(interpolated_df_2021)

## COMPUTE DAILY HOUR (0-23)
interpolated_df_2020 = compute_daily_hour_column(interpolated_df_2020)
interpolated_df_2021 = compute_daily_hour_column(interpolated_df_2021)

## COMPUTE TIME OF DAY (morning/day/afternoon/night)
interpolated_df_2020 = compute_time_of_day(interpolated_df_2020)
interpolated_df_2021 = compute_time_of_day(interpolated_df_2021)
