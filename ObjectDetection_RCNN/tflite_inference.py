import tensorflow as tf
from utils import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import imutils
import pickle
import time
import cv2
import os

testFile = config.TEST_PATHS

filetype = mimetypes.guess_type(testFile)[0]
imagePaths = [testFile]

# load the image paths in our testing file
imagePaths = open(testFile).read().strip().split("\n")

print("[INFO] loading object detector...")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

print(input_scale, input_zero_point)
print(output_scale, output_zero_point)

lb = pickle.loads(open(config.LB_PATH, "rb").read())

for (j, imagePath) in enumerate(imagePaths):
    # load the input image (in Keras format) from disk and preprocess
    # it, scaling the pixel intensities to the range [0, 1]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    # predict the bounding box of the object along with the class # label

    input_data = np.array(image, dtype=np.float32)
    print(input_data)
    input_data = input_data / input_scale + input_zero_point
    print(input_data)
    input_data = np.array(input_data, dtype=np.int8)
    print(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.perf_counter()

    interpreter.invoke()
    try:
        boxPreds = np.array(interpreter.get_tensor(output_details[0]['index']), dtype=np.int16)
        boxPreds = (boxPreds - output_zero_point) * output_scale
        print(boxPreds)

        inference_time = time.perf_counter() - start
        print("Success - inference time:  %.1fms" % (inference_time * 1000))
    except ValueError:
        print("Unsuccessful. Something went wrong")

    startX, startY, endX, endY = boxPreds[0]
    # determine the class label with the largest predicted probability

    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # scale the predicted bounding box coordinates based on the image # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    # draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY),
                  (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

