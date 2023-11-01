from keras.layers import Input,UpSampling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from qkeras import QConv2D
from keras import Sequential

# 定义数据路径
low_res_folder = './dataset'
high_res_folder= './h_test'

# 创建数据生成器
datagen = ImageDataGenerator()

# 加载数据
low_res_data = datagen.flow_from_directory(low_res_folder, target_size=(100, 100), class_mode=None, batch_size=32)
high_res_data = datagen.flow_from_directory(high_res_folder, target_size=(200, 200), class_mode=None, batch_size=32)

model=Sequential()
model.add(Input(shape=(100,100,3)))
model.add(Conv2D(16,3,padding='same', activation='relu'))
model.add(UpSampling2D((2,2),interpolation='nearest'))
model.add(Conv2D(3,3,padding='same', activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 打印训练数据的形状
print("X_train shape:", low_res_data)  # (样本数量, 图像高度, 图像宽度, 通道数)
print("y_train shape:", high_res_data)  # (样本数量, 图像高度 * 放大比例, 图像宽度 * 放大比例, 通道数)
# 训练模型
#model.fit(low_res_data, high_res_data, epochs=10, batch_size=30)

#model.save('haha.h5')