import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import matplotlib.pyplot as plt
import cv2

CATEGORIES = ["Cercospora", "Common_rust", "Healthy", "Northern_Leaf_Blight"]

# Load data training
x = np.load('x_train.npy')
y = np.load('y_train.npy')

x_train, x_validate, x_test = np.split(x, [int(.8*len(x)), int(.9*len(x))])
y_train, y_validate, y_test = np.split(y, [int(.8*len(y)), int(.9*len(y))])

# Data scaling
x_train = x_train.astype('float32')

x_train /= 255.0 - 0.5

# Creating a sequential model for CNN in Keras
num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential()
model.add(Conv2D(num_filters, filter_size, strides=(1, 1), input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, to_categorical(y_train), epochs=3, verbose=1,validation_data=(x_validate, to_categorical(y_validate)),)


score = model.evaluate(x_test, to_categorical(y_test), verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])


# Predict first 4 images
predictions = model.predict(x_test[:4])
# Print model's prediction
predictions_result = np.argmax(predictions, axis=1)
print(predictions_result)

# See first 4 images
for i in range(4):
    plt.imshow(cv2.cvtColor(x_test[i], cv2.COLOR_BGR2RGB))
    plt.title('Prediction: ' +  CATEGORIES[predictions_result[i]] + '  /  Actual: ' + CATEGORIES[y_test[i]])
    plt.show()

