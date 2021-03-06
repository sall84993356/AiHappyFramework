from Framework.Common.FileProcess import file_process
from Framework.FeatureProcessing.Standard import data_standard
from Framework.PracticeModule.DataFactory import data_factory
from Framework.FeatureProcessing.Explore import data_statistics
from Framework.Common.ImageVisualize import image_show

class processor():
    # 初始数据获取
    def get_data(self):
        digits = data_factory().get_digits()
        fe_data = digits.data
        label = digits.target
        # images = digits.images
        label_onehot = data_standard().one_hot(label.reshape((label.shape[0]),1))
        # print(label_onehot)
        # # 标准化
        # x_new = standard_info.StandardScaler()
        return fe_data, label_onehot

    # 特征数据加工
    def input_x(self, data_X):
        # 特征数据加工处理
        return data_X

    # 标签数据加工
    def input_y(self, label):
        # 标签数据加工处理
        return label


if __name__ == '__main__':
    fe_data, label = processor().get_data()
    dataAnalysis = data_statistics(fe_data, label)
    dataAnalysis.whole_info()
