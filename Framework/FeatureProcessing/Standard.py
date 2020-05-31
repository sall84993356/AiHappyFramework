from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


class data_standard:
    def __init__(self, feature_data, label_data):
        self.feature_data = feature_data
        self.label_data = label_data

    def StandardScaler(self):
        esti = StandardScaler()
        esti.fit(self.feature_data)
        new_data = esti.fit_transform(self.feature_data, self.label_data)
        # print(new_data.shape)
        return new_data

    def MinMaxScaler(self):
        esti = MinMaxScaler()
        esti.fit(self.feature_data, self.label_data)
        new_data = MinMaxScaler().fit_transform(self.feature_data,
                                                self.label_data)
        # print(new_data.shape)
        return new_data

    def Normalizer(self):
        esti = Normalizer()
        esti.fit(self.feature_data)
        new_data = Normalizer().fit_transform(self.feature_data)
        # print(new_data.shape)
        # print(new_data)
        return new_data