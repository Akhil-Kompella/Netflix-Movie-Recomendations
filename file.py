import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Covid_Data.csv')
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values
print(X)
print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer.fit(X[:, 0:1])
X[: , 0:1] = imputer.transform(X[:, 0:1])
print(X)

imputer.fit(X[:, 4:5])
X[: , 4:5] = imputer.transform(X[:, 4:5])
print(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 6:] = sc.fit_transform(X_train[:, 6:])
X_test[:, 6:] = sc.fit_transform(X_test[:, 6:])
print(X_train)
