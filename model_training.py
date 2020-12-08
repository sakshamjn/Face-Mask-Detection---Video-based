from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to output face mask detector model")
    args = vars(ap.parse_args())
    return args


def get_images(path):
    data = []
    labels = []
    for imagePath in path:
        label = imagePath.split(os.path.sep)[-2]
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels


def one_hot_encoder(lb, labels):
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    return labels


def data_splitting(data, labels):
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)
    return trainX, testX, trainY, testY


def augmetation_image():
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    return aug


def model_construction(aug, xtrain, xtest, ytrain, ytest):
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # freezing all layers (so they wont be trained in first training process)
    for layer in baseModel.layers:
        layer.trainable = False
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] training head...")
    H = model.fit(
        aug.flow(xtrain, ytrain, batch_size=BS),
        steps_per_epoch=len(xtrain) // BS,
        validation_data=(xtest, ytest),
        validation_steps=len(xtest) // BS,
        epochs=EPOCHS)
    print("[INFO] evaluating network...")
    pred = model.predict(xtest, batch_size=BS)
    pred = np.argmax(pred, axis=1)
    print(classification_report(ytest.argmax(axis=1), pred,
                                target_names=lb.classes_))
    print("[INFO] saving mask detector model...")
    model.save(args["model"], save_format="h5")
    return model, H


def training_summary(H):
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


if __name__ == '__main__':
    INIT_LR = 1e-4
    EPOCHS = 20
    BS = 32
    args = argument_parser()
    lb = LabelBinarizer()
    path = list(paths.list_images(args["dataset"]))
    data, labels = get_images(path)
    labels = one_hot_encoder(lb, labels)
    aug = augmetation_image()
    xtrain, xtest, ytrain, ytest = data_splitting(data, labels)
    model, H = model_construction(aug, xtrain, xtest, ytrain, ytest)
    training_summary(H)
