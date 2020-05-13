import pickle
import os
import codecs
import jieba
from sklearn.externals import joblib

class file_process():
    
    def save_model_group(self, file, data_list):
        joblib.dump(data_list, file)

    def open_model_group(self, file):
        back_group = joblib.load(file)
        return back_group

    def save_feature_label(self, file, feature_data, label_data):
        save_file = open(file, "wb")
        pickle.dump(feature_data, save_file)  #顺序存入变量
        pickle.dump(label_data, save_file)
        print('save success')
        save_file.close()

    def open_feature_label(self, file):
        try:
            load_file = open(file, "rb")
            feature_data = pickle.load(load_file)  #顺序导出变量
            label_data = pickle.load(load_file)
            print('open success')
            return feature_data, label_data
        except IOError:
            load_file.close()
            return None, None
        except EOFError:
            load_file.close()
            return None, None
        else:
            load_file.close()
            return None, None

    def get_file_list(self, dirs):
        """
        dirs:输入文件夹
        output：获取文件夹中所有文件和文件名
        """
        file_names = []
        names = []
        for root, dirs, files in os.walk(dirs):
            for file in files:
                file_names.append(root + "/" + file)
                names.append(file.split("_")[0])
        return file_names, names

    def get_words(self, dirs):
        """
        dirs:输入文件夹
        out：获取分词结果，并进行保存
        """
        file_names, file_class = self.get_file_list(dirs)
        words = []
        print("Seging...")
        for itr in file_names:
            #解决编码问题
            with codecs.open(itr, "r", encoding="utf-8") as file_handle:
                txt = file_handle.read()
                seg_list = jieba.cut(txt, cut_all=False)
                words.append(" ".join(seg_list))
        return words, file_class
