# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.datasets import mnist
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

class Config:
    # specify the shape of the inputs for our network
    IMG_SHAPE = (240, 320, 1)

    # specify the batch size and number of epochs
    BATCH_SIZE = 64
    EPOCHS = 100

    # define the path to the base output directory
    BASE_OUTPUT = "output"

    # use the base output path to derive the path to the serialized
    # model along with training history plot
    MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
    PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# instantiate the config class
config = Config()

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors

	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)

	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
 
def build_siamese_model(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)

	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)

	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)

	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)

	# build the model
	model = Model(inputs, outputs)

	# return the model to the calling function
	return model

def make_pairs(images):
    num_images = len(images)

    # Initialize lists to hold pairs of images and labels
    pair_images = []
    pair_labels = []

    # Create pairs of similar images (using the same image)
    for i in range(num_images):
        # Select a random image index
        idx = random.randint(0, num_images - 1)
        # Append the same image twice to create a pair
        pair_images.append([images[idx], images[idx]])
        pair_labels.append([1])  # Label 1 for similar pair

    # Create pairs of dissimilar images (using different images)
    for i in range(num_images):
        # Select two random image indices
        idx1, idx2 = random.sample(range(num_images), 2)
        # Append the pair of randomly selected images
        pair_images.append([images[idx1], images[idx2]])
        pair_labels.append([0])  # Label 0 for dissimilar pair

    # Convert lists to numpy arrays
    pair_images = np.array(pair_images, dtype=object)
    pair_labels = np.array(pair_labels)

    return pair_images, pair_labels

def load_images_from_dir(directory, max_images=500, target_size=(240, 320)):
    images = []
    count = 0  # To keep track of how many images have been processed

    for filename in os.listdir(directory):
        if count >= max_images:  # Stop if we have loaded max_images
            break
        if filename.endswith('.png'):  # Assuming images are PNG files
            img_path = os.path.join(directory, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # Resize the image
            image = cv2.resize(image, (target_size[1], target_size[0]))  # cv2.resize expects (width, height)
            # Normalize the pixel values to range [0, 1]
            image = image / 255.0
            images.append(image)
            count += 1

    return np.array(images)
# Directory containing your unlabeled images
image_dir = 'Directory/path'

# Load and normalize the images
trainX = load_images_from_dir(image_dir)
testX = load_images_from_dir(image_dir)
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
(pairTrain, labelTrain) = make_pairs(trainX)
(pairTest, labelTest) = make_pairs(testX)
print("this is done")

# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
print("model is done")

# compile the model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])
pairTrain = pairTrain.astype('float32')
labelTrain = labelTrain.astype('float32')
pairTest = pairTest.astype('float32')
labelTest = labelTest.astype('float32')

# train the model
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE, 
	epochs=config.EPOCHS)

# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
plot_training(history, config.PLOT_PATH)