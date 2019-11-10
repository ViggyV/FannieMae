import pandas as pd
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Input
import xgboost
import sklearn

import pandas as pd
import sys, math, csv, os

LAT_COLUMN = 19
LNG_COLUMN = 20

TEMP_COLUMN = 6
DEW_COLUMN = 7
HUMID_COLUMN = 8
WEATHER_COLUMN = 16
WIND_COLUMN = 10

f = open("Stations.csv", "r")
closest = None
distance = 100000000

latitude = 32.9858
longitude = 96.7501 

f.readline()

for line in csv.reader(f.readlines()):
	lat, lng = (float(line[LAT_COLUMN])-latitude), (float(line[LNG_COLUMN])-longitude)
	dist = math.sqrt(lat*lat + lng*lng)
	
	if dist < distance:
		distance = dist
		closest = line

f.close()

toReturn = [TEMP_COLUMN, DEW_COLUMN, HUMID_COLUMN, WIND_COLUMN]

print([float(closest[item]) for item in toReturn])

save = True
with open('MODIF_DATA.csv', 'r') as f:
    table = pd.read_csv(f, names=['Time', 'Occupancy', 'Occupancy1', 'Occupancy2', 'IndoorTemp', 'IndoorHumid', 'IndoorAir', 'IndoorMean', 'IndoorCO2', 'OutdoorTemp', 'OutdoorHumid', 'OutdoorAir', 'Office', 'Floor', 'Location', 'FanClass', 'FanState', 'WindowState', 'CurrentThermoCool', 'BaseThermoCool', 'CurrentThermoHeat', 'BaseThermoHeat'])
    print(table.describe())
    del table['Time']
    del table['Occupancy1']
    del table['Occupancy2']
    del table['IndoorAir']
    del table['OutdoorAir']
    del table['IndoorMean']
    del table['IndoorCO2']
    del table['Office']
    del table['Floor']
    del table['Location']
    del table['BaseThermoCool']
    del table['BaseThermoHeat']
    hasFan = table.loc[table['FanState'].notnull()]
    Y = hasFan.loc[:, table.columns == 'FanState']
    X = hasFan.loc[:, table.columns != 'FanState']
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, stratify=Y, train_size=0.8, random_state=1)
    clf = xgboost.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', num_class=2, objective='multi:softmax', n_estimators=2500, verbosity=2)
    if not save:
        clf.load_model('weather.model')
        print(X_train.values[0])
        print(clf.predict(X_train.values[0]))
        
    else:
        clf.fit(X_train.values, Y_train.values)
        print(sklearn.metrics.classification_report(Y_test.values, clf.predict(X_test.values)))
        clf.save_model('modified.model')