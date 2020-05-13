from sklearn.model_selection import train_test_split,KFold

class data_split:
    def __init__(self, feature_data, label_data):
        self.feature_data = feature_data
        self.label_data = label_data

    def RandomSplit(self,test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.feature_data,
                                                            self.label_data,
                                                            test_size=test_size,
                                                            random_state=0)
        return X_train, X_test, y_train, y_test
    def KFold(self):
        X_train=[] 
        X_test=[] 
        y_train=[] 
        y_test=[]
        kf = KFold(n_splits=6,shuffle=True,random_state=0)
        for train_index, test_index in kf.split(self.feature_data):
            print('train_index', train_index, 'test_index', test_index)
        for index in train_index:
            X_train.append(self.feature_data[index])
            y_train.append(self.label_data[index])
        for index in test_index:
            X_test.append(self.feature_data[index])
            y_test.append(self.label_data[index])
        return X_train, X_test, y_train, y_test