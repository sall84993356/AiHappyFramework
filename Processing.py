from Framework.Common.FileProcess import file_process
class Processor():
    
    # 初始数据获取
    def get_data(self):
        fp=file_process()
        file=''
        fe_data,label_data=fp.open_feature_label(file)
        return fe_data,label_data
    
    # 特征数据加工
    def input_x(self, data_X):
        # 特征数据加工处理
        return data_X

    # 标签数据加工
    def input_y(self, label):
        # 标签数据加工处理
        return label 
    
