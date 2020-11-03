import tensorflow as tf
from TYY_utils import mk_dir, load_data_npz
from TYY_model import TYY_MobileNet_reg
#test_file = sys.argv[1]
#netType = int(sys.argv[2])

#image2, age2, image_size = load_data_npz(test_file)
#model = TYY_MobileNet_reg(image_size,alpha)()
model = TYY_MobileNet_reg(64,0.25)()
model_file = '../megaage_models/MobileNet/batch_size_50/mobilenet_reg_0.25_66/mobilenet_reg_0.25_66.h5'
model.load_weights(model_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tfmodel = converter.convert()
open ("model.tflite" , "wb") .write(tfmodel)
