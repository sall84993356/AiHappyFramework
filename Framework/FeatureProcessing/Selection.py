from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,RFE,SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class feature_selection:
    def __init__(self, feature_data, label_data):
        self.feature_data = feature_data
        self.label_data = label_data
    # 方差
    def Filter_VarianceThreshold(self, threshold_num):
        selector = VarianceThreshold(threshold=threshold_num).fit(
            self.feature_data, self.label_data)
        data = selector.transform(self.feature_data)
        print(data[0:5])
        print(selector.variances_)
        return data
    # 卡方
    def Filter_SelectKBest(self, k_num):
        print(self.label_data)
        selector = SelectKBest(chi2,
                               k=k_num).fit(self.feature_data,
                                            self.label_data.astype('int'))
        data = selector.transform(self.feature_data)
        print(data[0:5])
        print(selector.scores_)
        print('pvalues_:',selector.pvalues_)
        print(selector.get_support(indices=True))
        return data

    def Wrapper_RFE(self, k_num):
        selector = RFE(estimator=LogisticRegression(penalty='l2',C=5.0,solver='liblinear'),
                       n_features_to_select=k_num).fit(
                           self.feature_data, self.label_data.astype('int'))
        data = selector.transform(self.feature_data)
        print(data[0:5])
        # print(selector.ranking_)
        # print(data == self.feature_data)
        return data

    def Embedded_LR(self, c_num):
        selector = SelectFromModel(LogisticRegression(
            penalty="l1", C=0.1)).fit(self.feature_data,
                                      self.label_data.astype('int'))
        data = selector.transform(self.feature_data)
        print(data[0:5])
        # print(selector.scores)
        return data

    def Embedded_GBDT(self):
        selector = SelectFromModel(GradientBoostingClassifier()).fit(
            self.feature_data, self.label_data.astype('int'))
        data = selector.transform(self.feature_data)
        print(data[0:5])
        # print(selector.estimator_.feature_importances_)
        return data
    # 文本特征选择
    def Count_Vectorizer(self,data_text):
        selector = CountVectorizer(max_df=0.95,token_pattern=r"(?u)\b\w+\b", min_df=2,max_features=20000)
        data = selector.fit_transform(data_text)
        return data,selector
    # 文本降维LDA
    def Latent_Dirichlet_Allocation(self,data_text):
        selector = LatentDirichletAllocation(n_components=6)
        data = selector.fit_transform(data_text)
        return data,selector