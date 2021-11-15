import pickle

from utils import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
from scipy import io
import cv2
import os

def get_data():
    print("[INFO] loading dataset...")
    data = []
    labels = []
    bboxes = []
    imagePaths = []

    for matPath in paths.list_files(config.ANNOTS_PATH, validExts=(".mat")):

        label = matPath.split("\\")[-2]

        if label not in config.VALID_LABELS:
            continue

        mat_file = matPath
        mat = io.loadmat(mat_file)
        mat = {k: v for k, v in mat.items() if k[0] != '_'}

        startY, endY, startX, endX = mat.get('box_coord')[0]

        img_name = "image_" + matPath[-8:-4] + ".jpg"
        imgPath = os.path.join(config.IMAGES_PATH, label, img_name)

        image = cv2.imread(imgPath)

        (h, w) = image.shape[:2]

        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        image = load_img(imgPath, target_size=(224, 224))
        image = img_to_array(image)

        data.append(image)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imgPath)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    bboxes = np.array(bboxes, dtype="float32")
    imagePaths = np.array(imagePaths)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    if len(lb.classes_) == 2:
        labels = to_categorical(labels)

    split = train_test_split(data, labels, bboxes, imagePaths,
                             test_size=0.2, random_state=42)

    (trainImages, testImages) = split[:2]
    (trainLabels, testLabels) = split[2:4]
    (trainBboxes, testBboxes) = split[4:6]
    (trainPaths, testPaths) = split[6:]

    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(testPaths))
    f.close()

    #save label binarizer to disk

    f = open(config.LB_PATH, "wb")
    f.write(pickle.dumps(lb))
    f.close()

    return (trainImages, testImages), (trainLabels, testLabels), \
           (trainBboxes, testBboxes), (trainPaths, testPaths)