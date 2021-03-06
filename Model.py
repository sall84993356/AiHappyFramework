import os
import numpy as np
from Framework.Common.FileProcess import file_process


class globalData:
    current_work_dir = os.path.dirname(__file__)
    fileName = current_work_dir + '/data/model.pkl'

class model:
    def __init__(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def save_model(self, model_info):
        file_process().save_model_group(globalData.fileName, model_info)
        print('save success')

    def get_model(self):
        return file_process().open_model_group(globalData.fileName)

    def predict(self, data):
        method = self.get_model()
        # 使用训练好模型进行预测
        y_predict = method.predict(data)
        return y_predict

    def test_all(self):
        method = self.get_model()
        # 使用训练好模型进行预测
        y_predict = method.predict(self.x_test)
        print("TEST acc:", len(y_predict & self.y_test) / len(self.y_test))
        
