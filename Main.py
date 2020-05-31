import numpy as np
from Framework.FeatureProcessing.DataSet import DataSet
from Framework.Module.Classification import Classification

from Model import Model
# 设置训练参数
batch_size = 10
epoch = 15

# 初始化训练数据集
dataSet = DataSet(batch_size, epoch)

# 获取测试数据
x_test, y_test = dataSet.fetch_next_test()
model = Model(x_test, y_test)

# 查看数据空间分布
dataSet.VisualizeSourceData('scatter_3d')

# 初始化分类器
class_function = Classification()

# 一次训练
# 获取全量训练数据
X_train, Y_train = dataSet.fetch_all_data()
# 使用模型进行训练，可以切换不同的模型，并对比训练结果
method = class_function.train_logistic_regression(X_train, Y_train)
# method = class_function.train_knn(X_train, Y_train,5)
# 保存模型
model.save_model(method)
# 测试集得分
model.test_all()

# 批次训练 目前不支持持续学习
# for step in range(dataSet.get_step()):
#     # 获取训练数据
#     X_train, Y_train = dataSet.fetch_next_batch()

#     method = class_function.train_gaussian_nb(X_train, Y_train)

#     #保存模型
#     model.save_model(method)

#     # 测试集得分
#     model.test_all()
