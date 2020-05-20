from Framework.Common.FileProcess import file_process
from Framework.PracticeModule.DataFactory import data_factory
class Processor():
    
    # 初始数据获取
    def get_data(self):
        iris=data_factory().get_iris()
        fe_data=iris.data,
        label=iris.target
        return fe_data[0],label
    
    # 特征数据加工
    def input_x(self, data_X):
        # 特征数据加工处理
        return data_X

    # 标签数据加工
    def input_y(self, label):
        # 标签数据加工处理
        return label 
    
if __name__ == '__main__':
    fe_data,label=Processor().get_data()
    print(fe_data)
    print(label)