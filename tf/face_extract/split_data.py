#!/usr/bin/python

import sys
import re
import math

SPLIT_FACTOR = 0.08
SPLIT_CHAR = '|'
MIN_SAMPLES =  100
MAX_SAMPLES = 100000
MAX_WRITE_SAMPLES = 100 
REWRITE_CLASS = True
CLASS_PATH_DEPTH = 2

# face specific params
MIN_ROLL = -20
MAX_ROLL = 20
MIN_PAN = -20
MAX_PAN = 20
MIN_TILT = -20
MAX_TILT = 20

def write_batch(file_1, file_2, batch):
    out1_items = int(len(batch) - (math.floor(len(batch) * SPLIT_FACTOR)))

    for i in batch[:out1_items]:
        out1.write(i)

    for i in batch[out1_items:]:
        out2.write(i)

if __name__ == "__main__":
    if len(sys.argv) <> 4:
        print("Usage: split_data.py <input_file> <out_train_file> <out_valid_file>")
        exit(0)

    input_file = sys.argv[1]
    out_train_file = sys.argv[2]
    out_valid_file = sys.argv[3]

    base_url_re = "(.*)/(.*\.(jpg|JPG))"
    batch = []
    last_folder = ""
    count = -1
    class_num = 0

    f = open(input_file)
    out1 = open(out_train_file, "w+")
    out2 = open(out_valid_file, "w+")

    for line in sorted(f.readlines()):

        if line.startswith('#'):
            continue

        tokens = line.split(SPLIT_CHAR)
        file_path = tokens[0]
        file_class = file_path.split('/')[CLASS_PATH_DEPTH]
        pan = float(tokens[1])
        roll = float(tokens[2])
        tilt = float(tokens[3])

        if (MIN_PAN <= pan <= MAX_PAN and
            MIN_ROLL <= roll <= MAX_ROLL and
            MIN_TILT <= tilt <= MAX_TILT):

            if REWRITE_CLASS == True:
                file_class = str(class_num)

            t = file_path + SPLIT_CHAR + file_class + '\n'
            batch.append(t)

            r = re.search(base_url_re, tokens[0])
            current_folder = r.group(1) # eg ./train_data/Abhishek\ Bachan/crop
            if count <> -1:
                if current_folder == last_folder:
                    count += 1
                else:
                    if MIN_SAMPLES < count < MAX_SAMPLES:
                        print("%s: %s samples, limiting to %s" % (last_folder, count, MAX_WRITE_SAMPLES))
                        write_batch(out1, out2, batch[:MAX_WRITE_SAMPLES])
                        class_num += 1

                    last_folder = current_folder
                    batch = []
                    count = 1
            else:
                last_folder = current_folder
                count = 1

    if MIN_SAMPLES < count < MAX_SAMPLES:
        print("%s: %s" % (last_folder, count))
        write_batch(out1, out2, batch)

    f.close()
    out1.close()
    out2.close()

