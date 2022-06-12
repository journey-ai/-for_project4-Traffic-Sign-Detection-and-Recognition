
import csv
import warnings # Control warning messages that pop up
warnings.filterwarnings("ignore") # Ignore all warnings
import cv2
import cv2 as cv
import matplotlib.pyplot as plt # Plotting library
import matplotlib.image as mpimg
import numpy as np # Scientific computing library 
import pandas as pd # Library for data analysis
import pickle # Converts an object into a character stream (i.e. serialization)
import random # Pseudo-random number generator library
from sklearn.model_selection import train_test_split # Split data into subsets
from sklearn.utils import shuffle # Machine learning library
from subprocess import check_output # Enables you to run a subprocess
import tensorflow as tf # Machine learning library
from tensorflow import keras # Deep learning library
from tensorflow.keras import layers # Handles layers in the neural network
from tensorflow.keras.models import load_model # Loads a trained neural network
from tensorflow.keras.utils import plot_model # Get neural network architecture
 

model = load_model('./road_sign2.1.h5')

# first cell data
labelNumbers = []

# second cell data
labelNames = []

# open file for reading
with open('./german-traffic-signs/signnames.csv') as csvDataFile:

    # open file as csv file
    csvReader = csv.reader(csvDataFile)

    # loop over rows
    for row in csvReader:

        # add cell [0] to list of dates
        labelNumbers.append(row[0])

        # add cell [1] to list of scores
        labelNames.append(row[1])

# output data
# print(labelNumbers)
# print(labelNames)

def gray_scale(img):
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    return img
def equalize(img):
    img = cv.equalizeHist(img)
    return img
def normalize(img):
    img = img/255
    return img
def preprocess(img):
    img = gray_scale(img)
    img = equalize(img)
    img = normalize(img)
    return img

# testing model on a picture got from web (a real example)
# url = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fcdn-01.media-brady.com%2Fstore%2Fstuk%2Fmedia%2Fcatalog%2Fproduct%2Fcache%2F3%2Fimage%2F85e4522595efc69f496374d01ef2bf13%2F1563981677%2Fd%2Fm%2Fdmeu_rms22_1_std.lang.all.gif&f=1&nofb=1'
# raw_img = requests.get(url,stream=True)
# img = Image.open(raw_img.raw)
img = cv2.imread('images/t1.jpg')
plt.imshow(img)
plt.show()



img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img)
# print(img.shape)

img = cv.resize(img, (32, 32))
img = preprocess(img)
plt.imshow(img, cmap = plt.get_cmap('gray'))
plt.show()
# print(img.shape)
img = img.reshape(1, 32, 32, 1)
# print(img.shape)
# print("predicted sign: "+ str(model.predict_classes(img)))
index1 = int(model.predict_classes(img))

print("m1")
print("predicted sign: "+ str(model.predict_classes(img)))
print("predicted sign: "+ labelNames[index1+1])

probabilities =np.round_(model.predict(img)*100, decimals = 3)
probability = np.max(probabilities) 
probabilityIndex = np.argmax(probabilities)
print("predicted sign: "+ str(labelNames[probabilityIndex+1]))
print("predicted sign prob: "+ str(probability))
print(str(probabilities))

