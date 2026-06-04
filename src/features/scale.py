from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_scaler(X_train: np.ndarray):
    sc = StandardScaler()
    sc.fit(X_train)
    return sc
