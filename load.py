from parse import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import pickle
from concurrent.futures import ThreadPoolExecutor
import timeit
from nn import *


def load_data(clean=["Lot1/Clean", "Lot2/Clean"], dirty=["Lot1/Spam", "Lot2/Spam"]):
    data = []
    label = []
    for path in clean:
        for file in os.listdir(path):
            data.append(read_file(path + "/" + file))
            label.append(0)

    for path in dirty:
        for file in os.listdir(path):
            data.append(read_file(path + "/" + file))
            label.append(1)

    return data, label


with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    data = load_data()
    data = pd.DataFrame(list(zip(data[0], data[1])))
    data = data.sample(frac=1).reset_index(drop=True)
    sequences = list(map(lambda x: x if len(x) < max_length else x[:max_length], tokenizer.texts_to_sequences(data[0])))
    padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=max_length, padding='post')

    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Load the previously saved weights
    model = tf.keras.models.load_model('model.h5')

    model.load_weights(latest)
    predictions = model.predict(padded_sequences)

    test_loss, test_accuracy = model.evaluate(padded_sequences, data[1])

    model.save('model_in_train.h5')

    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_accuracy)

    # # # Re-evaluate the model
    # # loss, acc = model.evaluate(X_test, y_train, verbose=2)
    #
    # model.save('in_train_model.h5')
    #
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
