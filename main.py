import os
from email.header import decode_header
import email
import codecs

import numpy as np

from utils import *
from nn import *
import pickle

clean_path = "Clean"
spam_path = "Spam"
encoding_dict = {encoding: [] for encoding in ['iso-8859-15', 'windows-1252',
                                               'iso-8859-1', 'iso-8859-7', 'iso-2022-jp', 'koi8-r', 'utf-8']}

encodings = set()

import argparse


def main():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # define the -info argument
    parser.add_argument('-info', help='write project information to the specified output file')
    # Add the -scan argument
    parser.add_argument('-scan', nargs=2, metavar=('folder', 'output_file'))
    # parse the command line arguments
    args = parser.parse_args()

    # check if the -info argument was provided and write the project information to the output file
    if args.info:
        with open(args.info, 'w') as f:
            f.write('DjaDja detectorul\n')
            f.write('Milea Robert Stefan\n')
            f.write('duArms\n')
            f.write('1.0.6\n')

    # Check if the -scan argument was used
    if args.scan:
        folder = args.scan[0]
        output_file = args.scan[1]

        data = scan_folder(folder)
        model = tf.keras.models.load_model('model.h5')

        with open('tokenizer.pkl', 'rb') as f:
            with tf.device('/gpu:0'):
                tokenizer = pickle.load(f)
                sequences = tokenizer.texts_to_sequences(data)

                sequences = list(map(lambda x: x if len(x) < max_length else x[:max_length], sequences))
                padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=max_length, padding='post')
                predictions = model.predict(padded_sequences)

                test_loss, test_accuracy = model.evaluate(padded_sequences,
                    np.asarray([0] * len(padded_sequences))
                )

                print('Test loss: ', test_loss)
                print('Test accuracy: ', test_accuracy)

                with open(output_file, "w") as out:
                    for file_name, prediction in zip(os.listdir(folder), predictions):
                        file_name = file_name.replace(" ", "_")

                        res = "cln" if prediction[0] < 0.5 else "inf"
                        out.write(file_name + "|" + res + '\n')


if __name__ == '__main__':
    main()
