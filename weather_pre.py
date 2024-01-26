import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")

core_weather = pd.read_csv("core_weather.csv", index_col="date")
from sklearn.linear_model import Ridge

reg= Ridge(alpha=.1)
predictor=["precip","temp_avg","humidity"]
train=core_weather.loc[:"2018-12-31"]
test=core_weather.loc["2019-01-01":]
reg.fit(train[predictor], train["target"])

pickle.dump(reg,open('ridge_model.pkl','wb'))
# model=pickle.load(open('ridge_model.pkl','rb'))
#
# int_features= [86.0, 0, 0]
# feature_names = ['temp_avg', 'precip','humidity']
# predictor=["precip","temp_avg","humidity"]
# new_data_df = pd.DataFrame([int_features], columns=feature_names)
# prediction = model.predict(new_data_df[predictor])
# print(int_features)
# print(new_data_df)
# # prediction=model.predict_proba(final)
# print(f"prediction={prediction}")
