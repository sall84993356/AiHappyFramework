import numpy as np
from Framework.FeatureProcessing.DataSet import data_set
from Framework.Module.Classification import classification

from Model import model
# 设置训练参数
batch_size = 10
epoch = 15

# 初始化训练数据集
dataSet = data_set(batch_size, epoch)

# 获取测试数据
x_test, y_test = dataSet.fetch_next_test()
model = model(x_test, y_test)

# 初始化分类器
class_function = classification()

# 一次训练
# 获取全量训练数据
X_train, Y_train = dataSet.fetch_all_data()
# 使用模型进行训练，可以切换不同的模型，并对比训练结果
method = class_function.train_logistic_regression(X_train, Y_train)

# 保存模型
model.save_model(method)
# 测试集得分
model.test_all()

