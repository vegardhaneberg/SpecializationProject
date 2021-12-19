# Machine learning
import h2o
import catboost as cb
import seaborn as sns
import shap

from h2o.automl import H2OAutoML
from sklearn.linear_model import RANSACRegressor, HuberRegressor, LinearRegression, TheilSenRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

from Plot import incidence_and_ml_plot

## PARAMETERS USED IN MACHINE LEARNING ##
block_type = 'IIR'
inc_angles = [0, 20]
predictive_features_era5 = ['sr', 'time_of_day', 'sm']
predictive_features_smap = ['sr', 'time_of_day', 'smap_sm']
smap_boolean = True

## Select desired DataFrame based on incidence angle interval, and desired satellite block codes ##
filtered_df_2020 = interpolated_df_2020[interpolated_df_2020['block_code'] != block_type]
filtered_df_2020 = filtered_df_2020[filtered_df_2020['sp_inc_angle'] >= inc_angles[0]]
filtered_df_2020 = filtered_df_2020[filtered_df_2020['sp_inc_angle'] <= inc_angles[1]]

filtered_df_2021 = interpolated_df_2021[interpolated_df_2021['block_code'] != block_type]
filtered_df_2021 = filtered_df_2021[filtered_df_2021['sp_inc_angle'] >= inc_angles[0]]
filtered_df_2021 = filtered_df_2021[filtered_df_2021['sp_inc_angle'] <= inc_angles[1]]

## Select ancillary data to be used ##
if smap_boolean:
    filtered_df_2020 = filtered_df_2020[predictive_features_smap]
    filtered_df_2021 = filtered_df_2021[predictive_features_smap]
else:
    filtered_df_2020 = filtered_df_2020[predictive_features_era5]
    filtered_df_2021 = filtered_df_2021[predictive_features_era5]

############################################
### Huber Regression Prediction and Plot ###
############################################
inc_intervals = [[0, 20]]

for interval in inc_intervals:
    block_type = 'IIR'
    predictive_features_era5 = ['sr', 'time_of_day', 'sm']
    predictive_features_smap = ['sr', 'time_of_day', 'smap_sm']
    smap_boolean = True

    for i in range(2):
        
        filtered_df_2020 = interpolated_df_2020[interpolated_df_2020['block_code'] != block_type]
        filtered_df_2020 = filtered_df_2020[filtered_df_2020['sp_inc_angle'] >= interval[0]]
        filtered_df_2020 = filtered_df_2020[filtered_df_2020['sp_inc_angle'] <= interval[1]]

        filtered_df_2021 = interpolated_df_2021[interpolated_df_2021['block_code'] != block_type]
        filtered_df_2021 = filtered_df_2021[filtered_df_2021['sp_inc_angle'] >= interval[0]]
        filtered_df_2021 = filtered_df_2021[filtered_df_2021['sp_inc_angle'] <= interval[1]]
        
        if smap_boolean:
            filtered_df_2020 = filtered_df_2020[predictive_features_smap]
            filtered_df_2021 = filtered_df_2021[predictive_features_smap]
        else:
            filtered_df_2020 = filtered_df_2020[predictive_features_era5]
            filtered_df_2021 = filtered_df_2021[predictive_features_era5]
            
        X_train = filtered_df_2020[['time_of_day', 'sr']]
        X_test = filtered_df_2021[['time_of_day', 'sr']]

        if smap_boolean:
            y_train = filtered_df_2020['smap_sm']
            y_test = filtered_df_2021['smap_sm']
        else:
            y_train = filtered_df_2020['sm']
            y_test = filtered_df_2021['sm']

        X_train = pd.get_dummies(X_train, columns=['time_of_day'])
        X_test = pd.get_dummies(X_test, columns=['time_of_day'])

        model = HuberRegressor()
        #model = LinearRegression()
        #model = RANSACRegressor()
        #model = TheilSenRegressor()

        print('INC_ANGLE_INTERVAL: ', interval)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        print('MODEL: ', model)
        print('RMSE: ', rmse)

        z = np.polyfit(y_test, y_test, 1)
        p = np.poly1d(z)
        plt.plot(y_test,p(y_test),"r")
        plt.hist2d(y_test, preds, (70, 70), cmap=plt.cm.jet, norm=LogNorm(), cmin=1)
        
        if smap_boolean:
            path = '/Users/madsrindal/Desktop/Plots/HuberRegressionPlot[0-20]-SMAP.png'
            plt.title('Huber Regression SMAP SM', fontsize=18)
            plt.xlabel('True SMAP SM value [cm^3/cm^3]', fontsize=12)
            plt.ylabel('Predicted SM value [cm^3/cm^3]', fontsize=12)
        else:
            path = '/Users/madsrindal/Desktop/Plots/HuberRegressionPlot[0-20]-ERA5.png'
            plt.title('Huber Regression ERA5 SM', fontsize=18)
            plt.xlabel('True ERA5 SM value [m^3/m^3]', fontsize=12)
            plt.ylabel('Predicted SM value [m^3/m^3]', fontsize=12)
        
        bar = plt.colorbar()
        bar.ax.set_title('Density')
        # plt.savefig(path, bbox_inches='tight')
        smap_boolean=False
        plt.show()



################
### CATBOOST ###
################
X_train, y_train = filtered_df_2020.iloc[:, :-1], filtered_df_2020.iloc[:, -1]
X_test, y_test = filtered_df_2021.iloc[:, :-1], filtered_df_2021.iloc[:, -1]

cat_features_indices = np.where(X_train.dtypes != float)[0]
print('Categorical features on indices: ', cat_features_indices)
train_dataset = cb.Pool(X_train, y_train, cat_features=cat_features_indices) 
test_dataset = cb.Pool(X_test, y_test, cat_features=cat_features_indices)

model = cb.CatBoostRegressor(loss_function='RMSE')

grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
model.grid_search(grid, train_dataset)

pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print('Testing performance')
print('Incidence angle interval: ', inc_angles)
print('RMSE: ', rmse)
print('R2: ', r2)

## Plot CatBoost Feature Importance ##
sorted_feature_importance = model.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_feature_importance], 
        model.feature_importances_[sorted_feature_importance], 
        color='blue')
plt.title("CatBoost Feature Importance", fontsize=18)
plt.xlabel("Importance Percentage", fontsize=12)
#plt.savefig('/Users/madsrindal/Desktop/Plots/CatBoostFeatureimportance', bbox_inches='tight')
plt.show()

## Plot CatBoost Prediction Scatter Plot ##
plt.plot(y_test,p(y_test),"r")
plt.hist2d(y_test, pred, (70, 70), cmap=plt.cm.jet, norm=LogNorm(), cmin=1)
plt.title('CatBoost SMAP SM Predictions', fontsize=16)
plt.xlabel('True SMAP SM value [cm^3/cm^3]', fontsize=12)
plt.ylabel('Predicted SM value [cm^3/cm^3]', fontsize=12)
bar = plt.colorbar()
bar.ax.set_title('Density')
# plt.savefig('/Users/madsrindal/Desktop/Plots/CatBoostPredictions[0-20]-SMAP.png')
plt.show()


##################
### H2O AutoML ###
##################
h2o.init()

train_h2o = h2o.H2OFrame(filtered_df_2020)
test_h2o = h2o.H2OFrame(filtered_df_2021)

y = 'smap_sm'
x = train_h2o.columns
x.remove(y)
train_h2o['time_of_day'] = train_h2o['time_of_day'].asfactor()
test_h2o['time_of_day'] = test_h2o['time_of_day'].asfactor()

aml = H2OAutoML(balance_classes=False, max_models = 10, seed = 1)
aml.train(x = x, y = y, training_frame = train_h2o)
lb = aml.leaderboard

model1 = aml.get_best_model(algorithm="xgboost", criterion="rmse")
model2 = aml.get_best_model(algorithm="GBM", criterion="rmse")
model3 = aml.get_best_model(algorithm="DRF", criterion="rmse")
model4 = aml.get_best_model(criterion='rmse')

preds1 = model1.predict(test_h2o)
preds2 = model2.predict(test_h2o)
preds3 = model3.predict(test_h2o)
preds4 = model4.predict(test_h2o)

rmse_xgboost = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions1['predict'], squared=False)
rmse_gbm = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions2['predict'], squared=False)
rmse_drf = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions3['predict'], squared=False)
rmse_best = mean_squared_error(h2o.as_list(test_h2o['smap_sm'])['smap_sm'], predictions4['predict'], squared=False)

print('INC_ANGLE_INTERVAL: ', inc_angles)
print('RMSE XGBOOST: ', rmse_xgboost)
print('RMSE GBM: ', rmse_gbm)
print('RMSE DRF: ', rmse_drf)
print('RMSE BEST: ', rmse_best)