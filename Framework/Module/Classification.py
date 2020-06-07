from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from Framework.Module.MethodInfo import method_info


class classification:
    def __init__(self):
        self.coreMethod = method_info()
        print("Classification Init")

    def train_random_forest(self, x_train, y_train, depth, n_es):
        print('Classification:RandomForestClassifier')
        method = self.coreMethod.method if (
            self.coreMethod.method is not None) else RandomForestClassifier(
                criterion='entropy',
                max_depth=depth,
                n_estimators=n_es,
                oob_score=False)
        method.fit(x_train, y_train)
        print('train score:')
        print(method.score(x_train, y_train))
        self.coreMethod.set_mothod(method)
        return method

    def train_gradient_boosting_classifier(self, x_train, y_train):
        print('Classification:GradientBoostingClassifier')
        method = self.coreMethod.method if (
            self.coreMethod.method is not None
        ) else GradientBoostingClassifier()
        method.fit(x_train, y_train)
        self.coreMethod.set_mothod(method)
        return method

    def train_logistic_regression(self, x_train, y_train):
        print('Classification:LogisticRegression')
        method = self.coreMethod.method if (
            self.coreMethod.method is not None
        ) else linear_model.LogisticRegression(
            penalty='l2', multi_class='auto', solver='newton-cg')
        method.fit(x_train, y_train)
        print('train score:')
        print(method.score(x_train, y_train))
        self.coreMethod.set_mothod(method)
        return method

    def train_knn(self, x_train, y_train, neighbors_num):
        print('Classification:KNeighborsClassifier')
        print(self.coreMethod.method is not None)
        print(self.coreMethod.version)
        method = self.coreMethod.method if (
            self.coreMethod.method is not None
        ) else neighbors.KNeighborsClassifier(neighbors_num, weights='uniform')
        method.fit(x_train, y_train)
        print('train score:')
        print(method.score(x_train, y_train))
        self.coreMethod.set_mothod(method)
        return method

    def train_svm(self, x_train, y_train):
        print('Classification:SVM')
        print(self.coreMethod.method is not None)
        method = self.coreMethod.method if (
            self.coreMethod.method is not None) else svm.SVC(
                C=10.0,
                cache_size=200,
                class_weight=None,
                coef0=0.0,
                decision_function_shape=None,
                degree=3,
                gamma='auto',
                kernel='rbf',
                max_iter=-1,
                probability=False,
                random_state=None,
                shrinking=True,
                tol=0.001,
                verbose=False)
        method.fit(x_train, y_train)
        print('train score:')
        print(method.score(x_train, y_train))
        self.coreMethod.set_mothod(method)
        return method

    def train_gaussian_nb(self, x_train, y_train):
        print('Classification:GaussianNB')
        method = self.coreMethod.method if (
            self.coreMethod.method is not None) else GaussianNB()
        method.fit(x_train, y_train)
        print('train score:')
        print(method.score(x_train, y_train))
        self.coreMethod.set_mothod(method)
        return method

    def train_multinomial_nb(self, x_train, y_train):
        print('Classification:MultinomialNB')
        method = self.coreMethod.method if (
            self.coreMethod.method is not None) else MultinomialNB()
        method.fit(x_train, y_train)
        print('train score:')
        print(method.score(x_train, y_train))
        self.coreMethod.set_mothod(method)
        return method

    def train_bernoulli_nb(self, x_train, y_train):
        print('Classification:BernoulliNB')
        method = self.coreMethod.method if (
            self.coreMethod.method is not None) else BernoulliNB()
        method.fit(x_train, y_train)
        print('train score:')
        print(method.score(x_train, y_train))
        self.coreMethod.set_mothod(method)
        return method