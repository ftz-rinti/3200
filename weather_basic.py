import pandas as pd
import warnings
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

core_weather = pd.read_csv("core_weather.csv", index_col="date")

predictor=["precip","temp_avg","humidity"]
train=core_weather.loc[:"2018-12-31"]
test=core_weather.loc["2019-01-01":]

# Train Decision Tree Regression model
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(train[predictor], train["target"])

# # Make predictions on the test set
# y_pred_dt = dt_regressor.predict(test[predictor])


pickle.dump(dt_regressor,open('DT_model.pkl','wb'))
# model=pickle.load(open('ridge_model.pkl','rb'))
#
# int_features= [34.0, 0, 0]
# feature_names = ['temp_avg', 'precip','humidity']
# predictor=["precip","temp_avg","humidity"]
# new_data_df = pd.DataFrame([int_features], columns=feature_names)
# prediction = model.predict(new_data_df[predictor])
# print(int_features)
# print(new_data_df)
# # prediction=model.predict_proba(final)
# print(f"prediction={prediction}")



