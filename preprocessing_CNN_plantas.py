import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "./input_data"
CATEGORIES = ["Cercospora", "Common_rust", "Healthy", "Northern_Leaf_Blight"]

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass
    return training_data

training_data = create_training_data()
random.shuffle(training_data)

print(len(training_data))

x_train = []
y_train = []

for image, label in training_data:
    x_train.append(image)
    y_train.append(label)

# Reshaping images to work with Keras
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0], 256, 256, 3)
y_train = np.array(y_train)

# Saving our data
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
