import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# initialize and fit model
def knn_pred():
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(y_pred)

# calculate the mean error
    knn_mean_error = pd.DataFrame(np.abs(y_pred - y_test)).mean()
    print(knn_mean_error)


