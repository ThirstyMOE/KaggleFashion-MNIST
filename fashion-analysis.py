import tensorflow as tf
import pandas as pd
import numpy as np
import struct as st
import os
import h5py
# import matplotlib.pyplot as plt  # can't seem to import tkinter dependency

"""
A method that returns the train data and labels and the test data and labels
"""
def data_preprocessing():
    # Read in your pandas dataframes
    train_df = pd.read_csv("fashion-mnist_train.csv")
    test_df = pd.read_csv("fashion-mnist_test.csv")

    # Extract the labels column from your pandas dataframes
    train_labels = train_df.label
    test_labels = test_df.label
    # Extract the other pixel columns from your pandas dataframes
    train_data = train_df.drop(["label"], axis=1)
    test_data = test_df.drop(["label"], axis=1)

    # Convert from pandas dataframe to numpy array
    train_data = train_data.values
    test_data = test_data.values
    train_labels = train_labels.values
    test_labels = test_labels.values

    return train_data, train_labels, test_data, test_labels

def create_model_architecture():
    # Get the keras model. Feedforward model (Sequential)
    model = tf.keras.models.Sequential()

    # Use model.add() to add layers to the keras model
    # Flatten the data from 28 x 28 to 794? 1 dim tensor
    model.add(tf.keras.layers.Flatten())
    # Add hidden layers of 128 neurons with relu activation
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # Add output layer of 10 neurons with softmax activation (for probablistic distr.)
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model

# def extract_labels():  # Got image extraction
#     filename = {'images' : 'train-images-idx3-ubyte' ,'labels' : 'train-labels-idx1-ubyte'}
#     imagesfile = open(filename['images'],'rb')
#     imagesfile.seek(0) # train_imagesfile
#     magic = st.unpack('>4B',imagesfile.read(4))
#     nImg = st.unpack('>I',imagesfile.read(4))[0] #num of images
#     nR = st.unpack('>I',imagesfile.read(4))[0] #num of rows
#     nC = st.unpack('>I',imagesfile.read(4))[0] #num of column
#     images_array = np.zeros((nImg,nR,nC))
#     nBytesTotal = nImg*nR*nC*1 #since each pixel data is 1 byte
#     images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,imagesfile.read(nBytesTotal))).reshape((nImg,nR,nC))
#     return images_array
# # A method to read in ubytes binary as a generator
# def read(dataset="training", path = "."):
#     """
#     Python function for importing the MNIST data set.  It returns an iterator
#     of 2-tuples with the first element being the label and the second element
#     being a numpy.uint8 2D array of pixel data for the given image.
#     """
#     if dataset is "training":
#         fname_img = os.path.join(path, 'train-images-idx3-ubyte')
#         fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
#     elif dataset is "testing":
#         fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
#         fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
#     else:
#         print("Not training or testing")
#
#     # Load everything in some numpy arrays
#     with open(fname_lbl, 'rb') as flbl:
#         magic, num = st.unpack(">II", flbl.read(8))
#         lbl = np.fromfile(flbl, dtype=np.int8)
#
#     with open(fname_img, 'rb') as fimg:
#         magic, num, rows, cols = st.unpack(">IIII", fimg.read(16))
#         img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
#
#     get_img = lambda idx: (lbl[idx], img[idx])
#
#     # Create an iterator which returns each image in turn
#     for i in range(len(lbl)):
#         yield get_img(i)


# train_generator = read("training")
# test_generator = read("testing")





# Load in mnist data into training data and labels and test data and labels
x_train, y_train, x_test, y_test = data_preprocessing()

# Use keras to put all input data between 0 and 1. Normalization
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = create_model_architecture()

adam_optimizer = tf.keras.optimizers.Adam(lr=0.001)

# Choooose your dueling weapon. Optimizers, loss type, metrics for evaluation, regularization too I bet
model.compile(
    optimizer=adam_optimizer, # String name of optimizer or optimizer instance
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Train the model!
history = model.fit(x_train, y_train, epochs=3)

# Calculate the validation data
val_loss, val_acc = model.evaluate(x_test, y_test)
print("Validation loss: " + str(val_loss))
print("Validation accuracy: " + str(val_acc))

# Save your model weights and architecture! Checkpointing
model.save("tmoe_fashion_mnist.model")

# # Reload your trained model back into a keras model object
# new_model = tf.keras.models.load_model("tmoe_mnist.model")
# # only takes a [list], but have your model run through input data and spit out predictions list
# predictions = new_model.predict([x_test])
