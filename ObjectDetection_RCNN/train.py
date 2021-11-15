from utils import config
from utils.data_prep import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import visualkeras
import pickle
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

(trainImages, testImages), (trainLabels, testLabels), (trainBboxes, testBboxes), (trainPaths, testPaths) = get_data()


# --------- BUILDING THE MODEL ------------
#import the vgg16 net without the FC head
"""vgg = VGG16(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False
flatten = layers.Flatten()(vgg.output)"""

#import the mobilenet v2 without the FC head
mobilenet = MobileNetV2(weights='imagenet', include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))
mobilenet.trainable = False
flatten = layers.Flatten()(mobilenet.output)

# Attach the bbox regressor to the last feature maps yielded by the VGG
bboxHead = layers.Dense(128, activation="relu")(flatten)
bboxHead = layers.Dense(64, activation="relu")(bboxHead)
bboxHead = layers.Dense(32, activation="relu")(bboxHead)
bboxHead = layers.Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

# Attach the object classifier to the last feature maps yielded by the VGG
softmaxHead = layers.Dense(512, activation="relu")(flatten)
softmaxHead = layers.Dropout(0.5)(softmaxHead)
softmaxHead = layers.Dense(512, activation="relu")(softmaxHead)
softmaxHead = layers.Dropout(0.5)(softmaxHead)
softmaxHead = layers.Dense(65, activation="softmax",  # !!! instead of 65 -> len(lb.classes_)
                           name="class_label")(softmaxHead)

# Assemble the model (VGG body -> feature maps---> bounding box regressor (predictor))
#                    (      branches:        |___> object classifier)
model = Model(inputs=mobilenet.input, outputs=(bboxHead, softmaxHead))

color_map = defaultdict(dict)
color_map[layers.Conv2D]['fill'] = 'red'
color_map[layers.BatchNormalization]['fill'] = color_map[layers.Activation]['fill'] = \
                                         color_map[layers.InputLayer]['fill'] = 'white'
color_map[layers.MaxPooling2D]['fill'] = 'orange'
color_map[layers.Flatten]['fill'] = 'gray'
color_map[layers.Dense]['fill'] = 'green'
color_map[layers.Dropout]['fill'] = 'teal'

visualkeras.layered_view(model, spacing=50, color_map=color_map,
        legend=True).save(os.path.join(config.BASE_OUTPUT, "visualizations\\volumetric_model.png"))
plot_model(model, os.path.join(config.BASE_OUTPUT, "visualizations\\graph_model.png"), show_shapes=True)

# Configuring the model
losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "MSE"
}

lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

opt = Adam(learning_rate=config.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=['accuracy'], loss_weights=lossWeights)

# --------- TRAINING THE MODEL -----------
trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBboxes
}
testTargets = {
    "class_label": testLabels,
    "bounding_box": testBboxes
}

print("[INFO] training model...")
H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets),
              batch_size=config.BATCH_SIZE, epochs=config.NUM_EPOCHS, verbose=1)

print("[INFO] saving the object detector model...")
model.save(config.MODEL_PATH)

# Plotting the metrics
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

for (i, l) in enumerate(lossNames):
    title = "Loss for {}".format(l) if l != "loss" else "Total Loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()

plt.tight_layout()
plotPath = os.path.join(config.PLOTS_PATH, "losses.png")
plt.savefig(plotPath)
plt.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"], label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"], label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

plotPath = os.path.sep.join([config.PLOTS_PATH, "accuraciess.png"])
plt.savefig(plotPath)