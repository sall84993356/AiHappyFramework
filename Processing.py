import jieba
import time 
from Framework.Common.FileProcess import file_process
class Processor():
    
    # 初始数据获取
    def get_data(self):
        fp=file_process()
        file='./第4周/3作业/2.16-AIE26-史明浩/Homework/Data/save_data.bin'
        fe_data,label_data=fp.open_feature_label(file)

        if fe_data is None or len(fe_data)==0:
            data, label = fp.get_words('./第4周/3作业/2.16-AIE26-史明浩/Homework/Data/news')
            print('data size:')
            print(len(data))
            data_seg = []
            st = time.clock()
            for itr in data:
                data_seg.append(' '.join(list(jieba.cut(itr))))
            ed = time.clock()
            print("Seg time:", ed-st)

            label2int = {}
            int2label = {}
            for idx, itr in enumerate(set(label)):
                label2int[itr] = idx
                int2label[idx] = itr
            label = [label2int[itr] for itr in label]

            #保存标签数据
            file_int2label_path='./第4周/3作业/2.16-AIE26-史明浩/Homework/Data/int2label.bin'
            fp.save_model_group(file_int2label_path,int2label)

            # 保存已经处理的文件
            fp.save_feature_label(file,data_seg, label)
            fe_data=data_seg
            label_data=label

        return fe_data,label_data
    
    # 特征数据加工
    def input_x(self, data_X):
        # 特征数据加工处理
        return data_X

    # 标签数据加工
    def input_y(self, label):
        # 标签数据加工处理
        return label
    
