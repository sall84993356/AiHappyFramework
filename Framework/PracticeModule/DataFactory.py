from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_diabetes, load_boston, load_wine, load_linnerud


class data_factory():
    def __init__(self):
        print("DataFactory init")

    def get_iris(self):
        """
        获取鸢尾花数据集
        """
        return load_iris()

    def get_digits(self):
        """
        获取手写数字识别数据集
        """
        return load_digits()

    def get_breast_cancer(self):
        """
        获取乳腺癌数据集，简单经典的用于二分类任务的数据集
        """
        return load_breast_cancer()

    def get_diabetes(self):
        """
        获取糖尿病数据集
        """
        return load_diabetes()

    def get_wine(self):
        """
        获取红酒数据集
        """
        return load_wine()

    def get_linnerud(self):
        """
        获取体能训练数据集
        """
        return load_linnerud()
