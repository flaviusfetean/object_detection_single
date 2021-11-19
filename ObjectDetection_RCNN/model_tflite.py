import tensorflow as tf
from utils import config
import numpy as np
from tensorflow.keras.models import load_model
import cv2

imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = imagePaths[:400]

def representative_dataset():
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (224, 224))
        image = np.array(image).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        if len(image.shape) == 2:
            np.stack([image, image, image], axis=-1)

        if i % 5 == 0:
          print("[INFO] Successfully yielded {} images for inference".format(i))

        yield [image.astype(np.float32)]


model = load_model(config.MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
