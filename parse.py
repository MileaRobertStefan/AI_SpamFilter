import codecs
import email.header
import quopri
import email
from utils import *


def read_file(path):
    try:
        try:
            with open(path, "r") as file:
                msg = email.message_from_file(file)
        except:
            with open(path, "rb") as file:
                msg = email.message_from_binary_file(file)
        headers = "".join(msg.values())

        try:
            body = msg.as_bytes()
        except UnicodeEncodeError:
            body = codecs.encode(msg.as_string(), "utf-8")

        body = body[body.find(msg.values()[-1].encode("utf-8")) + len(msg.values()[-1]):].strip()

        for (decoded, encoding) in decode_header(headers):
            if encoding:
                body = codecs.decode(body, encoding)

                break

        if type(body) == bytes:
            body = codecs.decode(body, "utf-8")

        body = body.strip()
        if body.count(" ") == 0:
            try:
                body = base64.b64decode(body)
                for codec in ["utf-8", "koi8-r"]:
                    try:
                        body = codecs.decode(body, codec)
                        break
                    except UnicodeDecodeError:
                        pass
            except ValueError:
                pass

        return body
    except Exception as e:

        # print(e)
        # print(path)
        pass

    return ""


if __name__ == "__main__":
    base_path = "Lot2/Spam"
    files = os.listdir(base_path)

    words = set()

    returned_bytes = 0
    returned_nothing = 0

    for file in files:
        try:
            a = read_file(base_path + "/" + file)
            if a == "":
                returned_nothing += 1

            words |= {type(a)}

            if type(a) == bytes:
                returned_bytes += 1
                print(file)

        except PermissionError:
            pass

    # print(read_file("Lot2/Spam/afdfbe1c7a0e49a26e8da7d2a36bf25d"))

    print(words)

    print("returned_bytes", returned_bytes)
    print("returned_nothing", returned_nothing)

    print_num_files(base_path)