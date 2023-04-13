import re
import os
from email.header import decode_header
import email
import codecs
import base64

clean_path = "Clean"
spam_path = "Spam"
encoding_dict = {encoding: [] for encoding in ['iso-8859-15', 'windows-1252',
                                               'iso-8859-1', 'iso-8859-7', 'iso-2022-jp', 'koi8-r', 'utf-8']}


def is_base64(string: str):
    string.replace("\n","")
    base64_regex = r'^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$'
    return bool(re.match(base64_regex, string))

def move_file(file_path, folder_path):
    # Make sure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Move the file into the folder
    file_name = os.path.basename(file_path)
    new_path = os.path.join(folder_path, file_name)
   
    print( os.rename(file_path, new_path))
    
def print_num_files(folder):
    # Get a list of all the files in the given folder
    files = os.listdir(folder)

    # Print the number of files
    print(f"There are {len(files)} files in the folder.")

