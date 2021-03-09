# FeatureImportances module did not work in jupyter notebook

from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.model_selection import FeatureImportances

features = ['valence', 'acousticness', 'danceability',
            'duration_ms', 'energy', 'explicit', 'year', 'instrumentalness', 'key',
            'liveness', 'loudness', 'mode', 'speechiness', 'tempo']
features_2 = ['valence', 'acousticness', 'danceability',
              'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key',
              'liveness', 'loudness', 'mode', 'speechiness', 'tempo']

data = pd.read_csv('./data/processed_data.csv')
data_without_old = pd.read_csv('./data/data_2000_2020.csv')

X_1 = data[features]
y_1 = data.popularity

X_2 = data_without_old[features]
y_2 = data_without_old.popularity

X_3 = data_without_old[features_2]
y_3 = data_without_old.popularity

train_X_1, test_X_1, train_y_1, test_y_1 = train_test_split(X_1, y_1, test_size=0.1, random_state=0)
train_X_2, test_X_2, train_y_2, test_y_2 = train_test_split(X_2, y_2, test_size=0.1, random_state=0)
train_X_3, test_X_3, train_y_3, test_y_3 = train_test_split(X_3, y_3, test_size=0.1, random_state=0)


rfr_model_1 = RandomForestRegressor()
feature_importance_1 = FeatureImportances(rfr_model_1)
rfr_model_1.fit(train_X_1, train_y_1)

rfr_model_2 = RandomForestRegressor()
feature_importance_2 = FeatureImportances(rfr_model_2)
rfr_model_2.fit(train_X_2, train_y_2)

rfr_model_3 = RandomForestRegressor()
feature_importance_3 = FeatureImportances(rfr_model_3)
rfr_model_3.fit(train_X_3, train_y_3)


feature_importance_1.show()
feature_importance_2.show()
feature_importance_3.show()
