import tensorflow as tf
import numpy as np
import pandas as pd

clothing_dict = {
"0": "T-shirt/top",
"1": "Trouser",
"2": "Pullover",
"3": "Dress",
"4": "Coat",
"5": "Sandal",
"6": "Shirt",
"7": "Sneaker",
"8": "Bag",
"9": "Ankle boot",
}

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

def cross_compare_answers(predictions, validation_labels):
    correct = 0.0
    total = 0.0
    for index in range(len(predictions)):
        # numpy's argmax to take the onehot array and bring out the strongest fired output
        predict_digit = str(np.argmax(predictions[index]))
        real_digit = str(validation_labels[index])
        total += 1
        if predict_digit == real_digit:
            correct += 1
        predict_digit, real_digit = translate_number_to_clothing(predict_digit, real_digit)
        print("Us: " + predict_digit + " -- Them: " + real_digit)
    print("You got an accuracy of: " + str(correct/total))

def translate_number_to_clothing(predict_digit, real_digit):
    # 0 T-shirt/top
    # 1 Trouser
    # 2 Pullover
    # 3 Dress
    # 4 Coat
    # 5 Sandal
    # 6 Shirt
    # 7 Sneaker
    # 8 Bag
    # 9 Ankle boot
    return clothing_dict[predict_digit], clothing_dict[real_digit]


x_train, y_train, x_test, y_test = data_preprocessing()

model = tf.keras.models.load_model("tmoe_fashion_mnist.model")

predictions = model.predict([x_test])

# Let's play a little game 1v1 me
cross_compare_answers(predictions, y_test)
