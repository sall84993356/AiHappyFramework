from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

class Classification:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def Train_RandomForest(self, depth, n_es,predict_data):
        print('Classification:RandomForestClassifier')
        method = RandomForestClassifier(criterion='entropy',
                                     max_depth=depth,
                                     n_estimators=n_es, oob_score=False)
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method

    def Train_GradientBoostingClassifier(self,predict_data):
        print('Classification:GradientBoostingClassifier')
        method = GradientBoostingClassifier()
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method

    def Train_LogisticRegression(self,predict_data):
        print('Classification:LogisticRegression')
        method = linear_model.LogisticRegression(penalty='l2', multi_class='auto',solver='newton-cg')
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method

    def Train_Knn(self,neighbors_num,predict_data):
        print('Classification:KNeighborsClassifier')
        method = neighbors.KNeighborsClassifier(neighbors_num, weights='uniform')
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method
    def train_SVM(self,predict_data):
        print('Classification:SVM')
        method = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method
    def train_GaussianNB(self,predict_data):
        print('Classification:GaussianNB')
        method = GaussianNB()
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method
    def train_MultinomialNB(self,predict_data):
        print('Classification:MultinomialNB')
        method = MultinomialNB()
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method
    def train_BernoulliNB(self,predict_data):
        print('Classification:BernoulliNB')
        method = BernoulliNB()
        method.fit(self.x_train, self.y_train)
        y = method.predict(predict_data)
        return y,method