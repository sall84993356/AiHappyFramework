import numpy as np
import jieba
from Framework.FeatureProcessing.DataSet import DataSet
from Framework.FeatureProcessing.Selection import feature_selection
from Framework.Training.Classification import Classification
from Model import Model
batch_size = 21600
epoch = 1
dataSet = DataSet(batch_size, epoch)
x_data,y_data=dataSet.fetch_next_test()
model=Model(x_data,y_data)
# 查看数据空间分布
# dataSet.VisualizeSourceData('scatter_3d');

# 初始化模型库
fs=feature_selection(None,None)

for step in range(dataSet.get_step()):
    # 获取训练数据
    X_train, label = dataSet.fetch_next_batch()
    print('X_train size:')
    print(len(X_train))
    print('-------------------------')

    tool_chain = []
    data_vect,method=fs.Count_Vectorizer(X_train)
    tool_chain.append(method)
    # print(len(data_vect))
    print('Count_Vectorizer process end')

    data_dm,method=fs.Latent_Dirichlet_Allocation(data_vect)
    tool_chain.append(method)
    print('Latent_Dirichlet_Allocation process end')

    cf=Classification(data_dm, label)
    y,method=cf.Train_RandomForest(100,50,data_dm)
    # y,method=cf.Train_GradientBoostingClassifier(data_dm)
    # y,method=cf.Train_LogisticRegression(data_dm)
    # y,method=cf.Train_Knn(7,data_dm)
    # y,method=cf.train_SVM(data_dm)
    # y,method=cf.train_GaussianNB(data_dm)
    # y,method=cf.train_MultinomialNB(data_dm)
    # y,method=cf.train_BernoulliNB(data_dm)
    tool_chain.append(method)
    print("RFC acc:", np.sum(y==label)/len(label))
    
    #保存模型
    model.save_model(tool_chain)
    
    # 测试集得分
    model.test_all()
    

