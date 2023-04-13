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

max_length = 500


def scan_folder(folder):
    data = []
    for file in os.listdir(folder):
        data.append(read_file(folder + "/" + file))

    return data

def new_scan_folder(folder):
    data = []
    for file in os.listdir(folder):
        data.append(read_file(folder + "/" + file))

    return data

def load_data(clean = ["Lot1/Clean", "Lot2/Clean"], dirty = ["Lot1/Spam", "Lot2/Spam"]):
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


def main():
    data = load_data()
    data = pd.DataFrame(list(zip(data[0], data[1])))
    data = data.sample(frac=1).reset_index(drop=True)

    data2 = load_data(["Lot2/Clean"],["Lot2/Spam"])
    data2 = pd.DataFrame(list(zip(data2[0], data2[1])))
    data2 = data2.sample(frac=1).reset_index(drop=True)
    # Tokenize the text
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data[0])
    sequences = tokenizer.texts_to_sequences(data[0])
    sequences2 = tokenizer.texts_to_sequences(data2[0])

    # Save the tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Get the word -> integer mapping
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')

    sequences = list(map(lambda x: x if len(x) < max_length else x[:max_length], sequences))
    sequences2 = list(map(lambda x: x if len(x) < max_length else x[:max_length], sequences2))

    padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=max_length, padding='post')
    padded_sequences2 = tf.keras.utils.pad_sequences(sequences2, maxlen=max_length, padding='post')

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data[1], test_size=0.001)

    # Define the model
    with tf.device('/gpu:0'):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(len(word_index) + 1, 32, input_length=max_length))
        model.add(tf.keras.layers.LSTM(units=8, return_sequences=True))
        model.add(tf.keras.layers.GRU(units=8, return_sequences=True))
        model.add(tf.keras.layers.LSTM(units=8))
        model.add(tf.keras.layers.Dense(units=4, activation='relu'))
        model.add(tf.keras.layers.Dense(units=4, activation='relu'))
        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=100 * 32)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Save the weights using the `checkpoint_path` format
        model.save_weights(checkpoint_path.format(epoch=0))

        # Train the model
        model.fit(X_train, y_train, epochs=1000, batch_size=32,
                  callbacks=[cp_callback]
                  )

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test,y_test)

        model.save('model.h5')

        print('Test loss: ', test_loss)
        print('Test accuracy: ', test_accuracy)


if __name__ == "__main__":
    main()

