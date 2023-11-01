from keras.layers import Input,UpSampling2D,Conv2D
from qkeras import QConv2D
from keras import Sequential
import numpy as np
from PIL import Image
import os

model=Sequential()
model.add(Input(shape=(100,100,3)))
model.add(Conv2D(16,3,padding='same', activation='relu'))
model.add(UpSampling2D((2,2),interpolation='bilinear'))
model.add(Conv2D(3,3,padding='same', activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

def prepare_data(low_res_folder, high_res_folder, image_size, scale, summ):
    low_res_images = []
    high_res_images = []
    i = 0
    j = 0
    # 遍历低分辨率图像文件夹
    for filename in os.listdir(low_res_folder):
        
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 加载低分辨率图像
            low_res_image = Image.open(os.path.join(low_res_folder, filename))

            # 调整图像大小
            low_res_image = low_res_image.resize(image_size)

            # 添加批次维度，并归一化
            low_res_image = np.array(low_res_image) / 256
            low_res_image = np.expand_dims(low_res_image, axis=0)

            low_res_images.append(low_res_image)
            i = i + 1
            if i == summ:
                break
    # 遍历高分辨率图像文件夹
    for filename in os.listdir(high_res_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 加载高分辨率图像
            high_res_image = Image.open(os.path.join(high_res_folder, filename))

            # 调整图像大小
            high_res_image = high_res_image.resize((image_size[0] * scale, image_size[1] * scale))

            # 添加批次维度，并归一化
            high_res_image = np.array(high_res_image) / 256
            high_res_image = np.expand_dims(high_res_image, axis=0)

            high_res_images.append(high_res_image)
            j = j + 1
            if j == summ:
                break

    # 将图像列表转换为NumPy数组并堆叠
    X_train = np.vstack(low_res_images)
    y_train = np.vstack(high_res_images)

    return X_train, y_train

low_res_folder = './dataset/l_test'
high_res_folder = './dataset/h_test'
image_size = (100, 100)  # 低分辨率图像的目标大小
scale = 2  # 放大比例

X_train, y_train = prepare_data(low_res_folder, high_res_folder, image_size, scale, 100)

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=30)

model.save('haha.h5')