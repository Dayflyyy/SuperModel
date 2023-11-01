import hls4ml
from keras.models import load_model

model=load_model("haha.h5")
hls_model=hls4ml.converters.convert_from_keras_model(model)