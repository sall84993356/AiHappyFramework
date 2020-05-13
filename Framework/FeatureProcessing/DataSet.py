import os
import sys
from ..Common.BasicVisualize import digram_show
from .DataSplit import data_split


class globalData():
    data_size = 0
    use_times = 0

class DataSet():
    def __init__(self, batch_size, epoch):
        print("DataSet init")
        self.batch_size = batch_size
        self.epoch = epoch
        self.DataInit()
        self.TrainTestSplit()

    # 初始化函数
    def DataInit(self):
        path = os.path.join(sys.path[0])
        print("test path:" + path)
        # 反射，调用processer模块，获取数据方法get_data
        moduleName = 'Processing'  # 要引入的模块
        className = "Processor"  # 要使用的方法
        model = __import__(moduleName, globals=path)  # 导入模块
        self.processClass = getattr(model, className)  # 找到模块中的属性

        self.X, self.y = self.processClass().get_data()

        globalData.data_size = len(self.X)
        print('DataSet init data')

    def TrainTestSplit(self):
        d_split = data_split(self.X, self.y)
        self.X, self.x_test, self.y, self.y_test = d_split.KFold()
        print('train test split :')
        print(len(self.X))
        print(len(self.x_test))

    # 训练初始数据可视化（简要）
    def VisualizeSourceData(self, show_type):
        X_data, y_data = self.X, self.y
        if show_type == 'plot':
            digram_show.show_plot(X_data, y_data)
        elif show_type == 'scatter':
            digram_show.show_scatter(X_data, y_data)
        elif show_type == 'plot_3d':
            print(type(X_data))
            digram_show.show_plot_3d([x[0] for x in X_data], y_data,
                                     [x[1] for x in X_data])
        elif show_type == 'scatter_3d':
            digram_show.show_scatter_3d([x[0] for x in X_data], y_data,
                                        [x[1] for x in X_data])

    # 获取一批训练数据
    def fetch_next_batch(self):
        if self.batch_size >= globalData.data_size:
            return self.data_process(self.X,self.y)
        else:
            # 获取数据的数量
            fetch_num = self.batch_size * globalData.use_times
            fetch_end_num = self.batch_size * (globalData.use_times + 1)
            # 处理X与y获取的批次数据
            batch_x = self.X[fetch_num:fetch_end_num]
            batch_y = self.y[fetch_num:fetch_end_num]
            globalData.use_times = globalData.use_times + 1
            return self.data_process(batch_x,batch_y)

    def data_process(self,x,y):
        x_data = self.processClass().input_x(x)
        y_data = self.processClass().input_y(y)
        return x_data,y_data

    def fetch_next_test(self):
        x_data,y_data=self.data_process(self.x_test, self.y_test)
        return x_data,y_data

    # 获取当前数据执行批次
    def get_step(self):
        return self.epoch
