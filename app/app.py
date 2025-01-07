import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
model = load_model('../models/skin_disease_model.h5')

# 数据加载与增强
train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=30, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory('../data/train', target_size=(224, 224), batch_size=32, class_mode='categorical')

# 预处理图像
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# 路由
@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Access the uploaded file
        file = request.files['file']

        # Ensure the uploads directory exists
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the file to the uploads directory
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Preprocess the image and make a prediction
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_name = list(train_generator.class_indices.keys())[class_index]

        # Render the result page
        return render_template('result.html', class_name=class_name, image_path=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

                      