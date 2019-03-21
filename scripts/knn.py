import app.preprocessing_tn as pre
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# load and clean housing data for knn-regression analysis
housing_data = pre.Preprocessing('housing')
housing_data.load_data('housing.csv', name='raw_data')
housing_data.cleanup('raw_data', drop=['total_bedrooms'], drop_duplicates=False, dropna={'axis': 1, 'thresh': 2})

# save cleaned data
housing_data.save('clean')
clean = housing_data.get('clean')

# set independend variable
X = clean.loc[:, 'longitude':'median_income']
print(X.isna().sum())

# set dependend variable
y = clean['median_house_value']

# split and normalize dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

# fit the knn model and predict mean value
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)

# calculate the mean error
error = pd.DataFrame(np.abs(y_pred - y_test)).mean()
print(error)

# use argparse command
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='knn')
    parser.add_argument('--filename', help='the file')

    parser.add_argument('--pre', default=False,
                        type=bool,
                        choices=(True, False),
                        help='running preprocessing')
    parser.add_argument('--train',
                        default=False,
                        type=bool,
                        choices=(True, False),
                        help='running training')

    args = parser.parse_args()

if args.pre:
    print('pre')

if args.train:
    print('train')
