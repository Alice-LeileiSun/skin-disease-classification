from sklearn.metrics import classification_report, confusion_matrix
#classification_report：生成分类任务的性能报告，包括精确率 (precision)、召回率 (recall)、
# F1 分数等。
#confusion_matrix：生成混淆矩阵，用于显示预测结果和真实标签的对比。
import seaborn as sns  
#导入 seaborn，一个用于数据可视化的 Python 库。在这里用于绘制混淆矩阵的热力图 (heatmap)。
import numpy as np 
#导入 numpy，一个用于科学计算的库。在这里用于对数组进行操作（如取最大值的索引）。
import matplotlib.pyplot as plt  
#导入 matplotlib.pyplot，一个用于生成各种图形的库。在这里用于显示混淆矩阵的热力图。
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

#加载测试数据
test_datagen = ImageDataGenerator(rescale=1.0/255)
#创建一个 ImageDataGenerator 对象，用于加载和预处理测试数据。
# rescale=1.0/255：将像素值从 [0, 255] 缩放到 [0, 1]，规范化数据，方便模型处理。
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224,224),
    batch_size=32, 
    class_mode='categorical',
    shuffle=False
)
#功能：从测试数据目录中加载图片，生成一个批量加载器 (test_generator)。
#'data/test'：测试数据的路径。
#target_size=(224,224)：将图片调整为 224x224 像素，以适配模型输入。
#batch_size=32：每次处理 32 张图片。
#class_mode='categorical'：将标签处理为分类格式（例如 [0,1,0,0] 表示第 2 类）。
#shuffle=False：不打乱数据顺序，保证预测结果与真实标签一一对应。

model = load_model('models/skin_disease_model.h5')

#预测
y_pred = model.predict(test_generator)
#功能：使用训练好的模型对测试数据进行预测，返回每张图片属于每个类别的概率。
#y_pred：是一个二维数组，每一行表示一张图片的类别概率。
y_pred_classes = np.argmax(y_pred, axis=1)
#功能：从预测结果中取出每张图片概率最大的类别的索引。
#np.argmax(y_pred, axis=1)：对每一行（每张图片）找出概率最大的类别索引。
#y_pred_classes：最终预测的类别标签（整数格式，例如 [0, 1, 2]）。
y_true = test_generator.classes  
#功能：获取测试数据的真实标签。
#test_generator.classes：包含每张图片的真实类别索引（与文件夹名称对应）。


#分类报告
print(classification_report(
    y_true,y_pred_classes, 
    target_names=test_generator.class_indices.keys())
)
#功能：生成并打印分类任务的性能报告。
#classification_report(y_true, y_pred_classes)：
#对比真实标签 y_true 和预测标签 y_pred_classes，
#计算每个类别的精确率、召回率、F1 分数和支持度 (support)。
#target_names=test_generator.class_indices.keys()：为类别索引添加对应的类别名称。

#混淆矩阵
cm = confusion_matrix(y_true,y_pred_classes)
#功能：计算混淆矩阵，用于展示每个类别的预测情况。
#混淆矩阵：一个二维表格，行表示真实类别，列表示预测类别，表格中的数字表示对应预测的次数。

sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=test_generator.class_indices.keys(), 
    yticklabels=test_generator.class_indices.keys()
)
#功能：使用 seaborn 绘制混淆矩阵的热力图。
#cm：混淆矩阵数据。
#annot=True：在每个单元格中显示数值。
#fmt='d'：数值格式为整数。
#cmap='Blues'：使用蓝色渐变的配色方案。
#xticklabels 和 yticklabels：为矩阵的横纵轴添加类别名称。
plt.title('Confusion Matrix')
plt.show()