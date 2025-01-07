import tensorflow as  tf 
 #This imports the TensorFlow library, which is an open-source platform for machine learning. It is used for building and training deep learning models.   
 # TensorFlow 是一个机器学习工具包，帮助我们用电脑教会程序自己学会识别东西，比如图片、声音等。就像你在学校里学知识，TensorFlow 就是一个“老师”，它有很多现成的课程和工具，帮你快速学会技能。
 
from tensorflow.keras.applications import EfficientNetB0
#tensorflow.keras.applications: A module in Keras (TensorFlow's high-level API) that provides pre-trained deep learning models for tasks like image classification and feature extraction.
#EfficientNetB0: A pre-trained convolutional neural network (CNN) architecture optimized for efficiency and performance. It's often used as a base model for transfer learning in image 
#classification tasks.
#这一句从 TensorFlow 的“高级功能库”里引入了一个叫 EfficientNetB0 的模型。EfficientNetB0 是一个“现成的模型”，就像一本学过的教科书，里面已经有很多知识（比如如何识别常见图片）。我们用它就可以“站在巨人的肩膀上”，
#直接利用它已经学会的知识，而不需要从零开始学习。

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
#These are common Keras layers used in building deep learning models:
#Dense:A fully connected layer, where each input neuron is connected to each output neuron.Used for classification or regression tasks.
#GlobalAveragePooling2D:A pooling layer that computes the average of all spatial features across the width and height of a feature map.
#Dropout:A regularization technique that randomly drops a fraction of neurons during training to prevent overfitting.
#这一句是说，“我要用一些搭建神经网络的零件”。你可以把这些看作搭建房子的材料：
#Dense（全连接层）：就像每根电线都连接到房子里的每个灯泡，它让每个输入的信息都参与决策，通常用于最后的输出结果，比如“这张图片是猫还是狗”。
#GlobalAveragePooling2D（全局平均池化）：这就像把大块的图片信息压缩成一个“总结”，但不会丢失重要的内容。用这个方法，电脑可以更快地理解图片。
#Dropout（丢弃层）：想象一下，你在考试前不看每一题的答案，而是随机挑一部分做练习。Dropout 的作用就是让模型变得更“健壮”（不会死记硬背），可以避免过度依赖某些特定信息，防止过拟合。

from tensorflow.keras.models import Model 
#Model:A class in Keras that represents a neural network model.You can define and compile custom models using the functional API. 
#这里的 Model 是用来“组装神经网络”的工具。它就像搭积木一样：你把所有零件（Dense、Dropout 等）拼在一起，最后形成一个完整的模型。比如说，你可以这样组装：图片 → 特征提取 → 分类结果（是猫还是狗）。

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#ImageDataGenerator:A utility for real-time data augmentation and preprocessing of image data.
#It can apply transformations like rotation, flipping, scaling, and normalization to increase the diversity of training data.
#这一句引入了一个叫 ImageDataGenerator 的工具，作用是对图片进行预处理和增强。它就像一个“图片加工厂”，可以给图片做以下处理：
   #缩放：把像素值调整到合理范围，比如 [0, 1]。
   #旋转：把图片随机旋转一定角度，比如左右转 20 度。
   #翻转：随机翻转图片，比如把图片水平翻转。
   #增强数据：相当于把原来的图片多做一些变化，让模型学得更好。
#这样做的好处是，即使只有很少的图片，也可以通过这些变化模拟出更多图片，增强模型的学习能力。


# Inclusion:
# EfficientNetB0:Provides a pre-trained feature extractor.
# Dense, GlobalAveragePooling2D, Dropout: #Build and customize the classification head on top of the pre-trained model.
# Model: Combines the base model and the classification head into a single neural network.
# ImageDataGenerator: Prepares and augments the image data for training and validation
#总的来说，这些代码是干什么的？
#这些代码是用来搭建一个“图片识别模型”的。步骤大致如下：
    #用 EfficientNetB0：直接利用一个已经学会了很多知识的模型。
    #加一些“自己的层”：用 Dense、Dropout 等，定义这个模型最终要学什么（比如识别猫和狗）。
    #处理图片：用 ImageDataGenerator 把图片调整成适合模型学习的样子。
    #组装模型：用 Model 把整个模型拼装好，最终可以用它来进行图片分类。



# 数据加载与增强
# 训练数据的增强：对图片进行缩放、旋转、随机缩放部分区域、水平翻转，增强数据多样性
train_datagen = ImageDataGenerator(
    rescale=1.0/255,          # 将像素值从 [0, 255] 缩放到 [0, 1]
    rotation_range=30,        # 随机旋转图片最多 30 度
    zoom_range=0.2,           # 随机缩放图片
    horizontal_flip=True      # 随机水平翻转图片
)

# 验证数据只进行缩放，不做其他增强
val_datagen = ImageDataGenerator(
    rescale=1.0/255           # 将像素值缩放到 [0, 1]
)

# 加载训练数据，自动从文件夹中加载并增强
train_generator = train_datagen.flow_from_directory(
    'data/train',             # 训练数据所在路径
    target_size=(224,224),    # 调整图片大小到 224x224（适配模型输入）
    batch_size=32,            # 每批次处理 32 张图片
    class_mode='categorical'  # 分类模式（多个类别），每张图片对应一个类别
)

# 加载验证数据
val_generator = val_datagen.flow_from_directory(
    'data/val',               # 验证数据所在路径
    target_size=(224,224),    # 调整图片大小到 224x224
    batch_size=32,            # 每批次处理 32 张图片
    class_mode='categorical'  # 多分类模式
)

# 加载预训练模型 EfficientNetB0
base_model = EfficientNetB0(
    weights='imagenet',       # 使用在 ImageNet 数据集上训练好的权重
    include_top=False,        # 不加载预训练模型的顶层分类部分，只用它提取特征
    input_shape=(224,224,3)   # 输入图片形状为 224x224x3（宽、高、RGB 通道）
)

# 冻结预训练模型的参数，训练时不会更新它的权重
base_model.trainable = False

# 自定义分类头
# 在预训练模型的输出基础上添加新的分类层
x = GlobalAveragePooling2D()(base_model.output)  # 全局平均池化，将特征压缩为一维
x = Dropout(0.2)(x)                              # 随机丢弃 20% 的神经元，防止过拟合
x = Dense(128, activation='relu')(x)             # 全连接层，128 个神经元，激活函数为 ReLU
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
# 输出层，神经元个数为类别数（train_generator.class_indices 自动获取类别数），使用 softmax 激活函数进行分类

# 创建完整模型
model = Model(
    inputs=base_model.input,   # 输入图片数据
    outputs=predictions        # 输出预测结果
)

# 编译模型
model.compile(
    optimizer='adam',                 # 使用 Adam 优化器（自动调整学习率）
    loss='categorical_crossentropy',  # 损失函数：多分类交叉熵
    metrics=['accuracy']              # 评估指标：分类准确率
)

# 训练模型
history = model.fit(
    train_generator,           # 使用训练数据
    validation_data=val_generator,  # 使用验证数据评估模型
    epochs=10                  # 训练 10 个轮次
)

# 保存模型到文件
model.save('models/skin_disease_model.h5')  # 将训练好的模型保存为 .h5 文件，便于后续加载和使用
