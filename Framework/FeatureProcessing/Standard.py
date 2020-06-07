from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class data_standard:
    def standard_scaler(self,feature_data, label_data):
        esti = StandardScaler()
        esti.fit(feature_data)
        new_data = esti.fit_transform(feature_data, label_data)
        return new_data

    def min_max_scaler(self,feature_data, label_data):
        esti = MinMaxScaler()
        esti.fit(feature_data, label_data)
        new_data = MinMaxScaler().fit_transform(feature_data, label_data)
        return new_data

    def normalizer(self,feature_data, label_data):
        esti = Normalizer()
        esti.fit(feature_data)
        new_data = Normalizer().fit_transform(feature_data)
        return new_data

    def one_hot(self, label_data):
        enc = OneHotEncoder(sparse=False)
        label_onehot= enc.fit_transform(label_data)
        return label_onehot.astype(np.uint8)

