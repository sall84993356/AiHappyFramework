import os
import sys
from .DataSplit import data_split


class globalData():
    data_size = 0
    use_times = 0


class data_set:
    def __init__(self, batch_size, epoch):
        print("data_set init")
        self.batch_size = batch_size
        self.epoch = epoch
        self.data_init()
        self.train_test_split()

    # 初始化函数
    def data_init(self):
        path = os.path.join(sys.path[0])
        print("test path:" + path)
        # 反射，调用processer模块，获取数据方法get_data
        moduleName = 'Processing'  # 要引入的模块
        className = "processor"  # 要使用的方法
        model = __import__(moduleName, globals=path)  # 导入模块
        self.process_class = getattr(model, className)  # 找到模块中的属性

        self.X, self.y = self.process_class().get_data()
        self.process_train_param()
        print('DataSet init data')

    def train_test_split(self):
        d_split = data_split(self.X, self.y)
        self.X, self.x_test, self.y, self.y_test = d_split.random_split(0.1)
        self.process_train_param()
        print('train test split :')
        print(self.y)
        print(self.y_test)

    def process_train_param(self):
        globalData.data_size = len(self.X)
        self.epoch = globalData.data_size // self.batch_size
        check_num = globalData.data_size % self.batch_size
        if check_num > 0:
            self.epoch += 1

    def fetch_all_data(self):
        return self.X, self.y

    # 获取一批训练数据
    def fetch_next_batch(self):
        if self.batch_size >= globalData.data_size:
            return self.data_process(self.X, self.y)
        else:
            # 获取数据的数量
            fetch_num = self.batch_size * globalData.use_times
            fetch_end_num = self.batch_size * (globalData.use_times + 1)

            # 处理X与y获取的批次数据
            batch_x = self.X[fetch_num:fetch_end_num]
            batch_y = self.y[fetch_num:fetch_end_num]
            globalData.use_times = globalData.use_times + 1
            return self.data_process(batch_x, batch_y)

    def data_process(self, x, y):
        x_data = self.process_class().input_x(x)
        y_data = self.process_class().input_y(y)
        return x_data, y_data

    def fetch_next_test(self):
        x_data, y_data = self.data_process(self.x_test, self.y_test)
        return x_data, y_data

    # 获取当前数据执行批次
    def get_step(self):
        return self.epoch
