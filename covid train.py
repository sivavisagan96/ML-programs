from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import keras
import numpy as np
import random
import pickle
import cv2
import os

# Parameters:-
EPOCHS = 100
##INIT_LR = 1e-3
##BS = 32
IMAGE_DIMS = (96, 96, 3)


# Input image:-
imagePaths = sorted(list(paths.list_images("Dataset")))
random.seed(42)# initialize the random number generator
random.shuffle(imagePaths)

# Create an list:-
data=[]
labels=[]

# loop over the input images
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(IMAGE_DIMS[1],IMAGE_DIMS[0]))
	
	image = img_to_array(image)
	data.append(image)

	l = label = imagePath.split(os.path.sep)[-2].split("_")
	
	labels.append(l)

# Convert into numpy both input & lables:-
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarizer implementation
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
print(labels.shape)
print(labels.ndim)
#  loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# Split Training & Testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# LENET ARCHITECTURE CODE

# initialize the model
model = Sequential()

# Rows & Columns  
imgRows=IMAGE_DIMS[0]
imgCols=IMAGE_DIMS[1]
numChannels=IMAGE_DIMS[2]
numClasses=2
inputShape = (imgRows, imgCols, numChannels)

activation="relu"
weightsPath=None

# define the first set of CONV => ACTIVATION => POOL layers
model.add(Conv2D(20, 5, padding="same",
        input_shape=inputShape))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the second set of CONV => ACTIVATION => POOL layers
model.add(Conv2D(50, 5, padding="same"))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# define the first FC => ACTIVATION layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation(activation))

# define the second FC layer
model.add(Dense(numClasses))

# lastly, define the soft-max classifier
model.add(Activation("softmax"))

# if a weights path is supplied (inicating that the model was
# pre-trained), then load the weights
if weightsPath is not None:
        model.load_weights(weightsPath)

#compile 
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = 'SGD',metrics = ['accuracy'])

# fitting the model 
hist = model.fit(x=trainX,y=trainY,epochs =EPOCHS ,batch_size = 128,
                 validation_data =(testX,testY),
                 verbose = 1)

# evaluate the model
test_score = model.evaluate(testX,testY)
print("Test loss {:.5f},accuracy {:.3f}".format(test_score[0],test_score[1]*100))

# Save the model
model.save('Model.h5')

f = open("MLB.PICKLE", "wb")
f.write(pickle.dumps(mlb))
f.close()

