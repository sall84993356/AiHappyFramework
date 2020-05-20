import numpy as np
from Framework.FeatureProcessing.DataSet import DataSet
from Framework.Module.Classification import Classification
from Model import Model
batch_size = 10
epoch = 15
dataSet = DataSet(batch_size, epoch)
x_data,y_data=dataSet.fetch_next_test()
model=Model(x_data,y_data)
# 查看数据空间分布
dataSet.VisualizeSourceData('scatter_3d')

# 初始化模型库
# fs=feature_selection(None,None)

for step in range(dataSet.get_step()):
    # 获取训练数据
    X_train, label = dataSet.fetch_next_batch()
    class_function=Classification(X_train, label)
    print(X_train)
    print(label)
    print(len(X_train))
    print(len(label))
    y,method= class_function.Train_Knn(3,x_data)
    # print('X_train size:')
    # print(len(X_train))
    # print('-------------------------')


    #保存模型
    model.save_model(method)
    
    # 测试集得分
    model.test_all()
    

