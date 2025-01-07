from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# Import functions for generating classification reports, confusion matrices, and ROC/AUC metrics
import seaborn as sns
# Import seaborn for visualizing data (e.g., heatmaps)
import numpy as np
# Import numpy for numerical operations
import matplotlib.pyplot as plt
# Import matplotlib for plotting graphs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import ImageDataGenerator for loading and preprocessing image data
from tensorflow.keras.models import load_model
# Import load_model to load the trained model
from sklearn.preprocessing import label_binarize
# Import label_binarize to convert class labels into binary format for ROC curves
import os
# Import os to check file existence and handle file paths

# 模型文件路径
model_path = 'models/skin_disease_model.h5'
# Define the file path for the saved model

# 检查模型文件是否存在
if not os.path.exists(model_path):
    # If the model file does not exist, raise an error
    raise FileNotFoundError(f"Model file not found: {model_path}. Please provide the correct path")

# 加载模型
model = load_model(model_path)
# Load the trained model from the specified file path

# 加载测试数据
test_datagen = ImageDataGenerator(rescale=1.0/255)
# Create an ImageDataGenerator to normalize pixel values to [0, 1]

test_generator = test_datagen.flow_from_directory(
    'data/test',                  # Path to the test data directory
    target_size=(224, 224),       # Resize images to 224x224 to match the model input
    batch_size=32,                # Number of images per batch
    class_mode='categorical',     # Treat the labels as categorical data (multi-class)
    shuffle=False                 # Ensure the order of data matches the order of predictions
)

# 预测
y_pred = model.predict(test_generator)
# Use the model to predict probabilities for the test data

y_pred_classes = np.argmax(y_pred, axis=1)
# Convert probabilities into class labels by selecting the class with the highest probability

y_true = test_generator.classes
# Get the true class labels from the test generator

# 分类报告
report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
# Generate a detailed classification report including precision, recall, F1 score, and support for each class

print(report)
# Print the classification report to the console

# 保存分类报告到文件
with open("evaluation_report.txt", "w") as f:
    # Open a file to save the classification report
    f.write(report)
    # Write the classification report to the file

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)
# Compute the confusion matrix to compare true labels with predicted labels

sns.heatmap(
    cm, 
    annot=True,                  # Annotate each cell with the corresponding number
    fmt='d',                     # Format the numbers as integers
    cmap='Blues',                # Use the 'Blues' colormap for the heatmap
    xticklabels=test_generator.class_indices.keys(),  # Set x-axis labels as class names
    yticklabels=test_generator.class_indices.keys()   # Set y-axis labels as class names
)
plt.title('Confusion Matrix')
# Set the title of the heatmap plot

plt.show()
# Display the confusion matrix plot

# 绘制 ROC 曲线
y_true_binary = label_binarize(y_true, classes=range(len(test_generator.class_indices)))
# Convert the true labels into a binary format (one-hot encoding) for each class

# 计算每个类别的 ROC 曲线和 AUC
plt.figure()
# Create a new figure for the ROC curves

for i in range(len(test_generator.class_indices)):
    # Loop through each class to calculate and plot its ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary[:, i], y_pred[:, i])
    # Compute the false positive rate (FPR) and true positive rate (TPR) for the class
    roc_auc = auc(fpr, tpr)
    # Calculate the area under the curve (AUC) for the class

    plt.plot(fpr, tpr, label=f"Class {list(test_generator.class_indices.keys())[i]} (AUC = {roc_auc:.2f})")
    # Plot the ROC curve and label it with the class name and AUC value

plt.plot([0, 1], [0, 1], 'k--')
# Plot the diagonal line representing a random classifier

plt.title("ROC Curve")
# Set the title for the ROC curve plot

plt.xlabel("False Positive Rate")
# Label the x-axis

plt.ylabel("True Positive Rate")
# Label the y-axis

plt.legend(loc="lower right")
# Add a legend to the lower-right corner of the plot

plt.show()
# Display the ROC curve plot
