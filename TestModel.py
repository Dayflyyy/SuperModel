import tensorflow as tf
from PIL import Image
import numpy as np

# 加载模型
model = tf.keras.models.load_model('F.h5')

# 加载测试图像
test_image = Image.open('./dataset/l_test/0.jpg')
test_image = test_image.resize((100, 100))  # 调整图像大小
test_image = np.array(test_image) / 256  # 归一化
test_image = np.expand_dims(test_image, axis=0)  # 添加批次维度

# 使用模型生成高分辨率图像
output_image = model.predict(test_image)

# 将输出图像从[0, 1]范围转换回[0, 255]范围
output_image = np.squeeze(output_image) * 256
output_image = output_image.astype(np.uint8)

# 保存生成的高分辨率图像
output_image = Image.fromarray(output_image)
output_image.save('output_image.jpg')