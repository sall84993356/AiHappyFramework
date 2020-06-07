from Model import Model
from Framework.Common.FileProcess import file_process

model = Model(None, None)
data = [6.7, 3.3, 5.7, 2.5]
result = model.predict(data)
print(result)
