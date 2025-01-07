import os
import shutil
import pandas as pd

#设置路径
csv_file = "data/raw/metadata.csv" # csv文件路径
images_folder = "data/raw/images" # 图片存放文件夹
output_folder = "data/raw" #输出的分类文件夹根目录

#读取csv文件
labels = pd.read_csv(csv_file)

#遍历 csv 中的每一行
for index,row in labels.iterrows():
    image_name = row['isic_id']
    image_name = image_name + '.jpg'
    category = str(row['diagnosis_3']) if not pd.isna(row['diagnosis_3']) else 'Unknown'
    
    #创建类别文件夹（如果不存在）
    category_folder = os.path.join(output_folder,category)
    os.makedirs(category_folder, exist_ok=True)
    
    #源图片路径
    src_path = os.path.join(images_folder,image_name)
    
    #目标路径
    dst_path = os.path.join(category_folder,image_name)
    
    #将图片移动到对应类别文件夹
    if os.path.exists(src_path):
        shutil.move(src_path,dst_path)
    else:
        print(f"Warning:{src_path} does not exist!")
    print("It is done")