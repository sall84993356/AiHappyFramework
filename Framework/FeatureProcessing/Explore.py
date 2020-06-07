import numpy as np
import pandas as pd


class data_statistics():
    def __init__(self, data_x, data_y):
        print('data_statistics.__init__')
        self.data_x = pd.DataFrame(data_x)
        self.data_y = pd.DataFrame(data_y)

    def whole_info(self):
        self.data_describe(self.data_x)
        self.data_describe(self.data_y)

    def data_describe(self, data_show):
        print('shape:')
        print(np.shape(data_show))
        print('-----------------------------')
        print('head:')
        print(data_show.head(5))  #显示前5行数据
        print('-----------------------------')
        print('tail:')
        print(data_show.tail(5))  #显示后5行
        print('-----------------------------')
        print('columns:')
        print(data_show.columns)  #查看列名
        print('-----------------------------')
        print('info:')
        print(data_show.info())  #查看各字段的信息
        print('-----------------------------')
        print('describe:')
        print(data_show.describe())  #查看数据的大体情况
        print('-----------------------------')


if __name__ == '__main__':
    print('')