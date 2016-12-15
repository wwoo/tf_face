import sys
import threading
import os
import socket
import urllib2
from Queue import Queue
from PIL import Image

NUM_THREADS = 4
URL_TIMEOUT = 4
IMAGE_CROP = False 
ESCAPE_SPACES = False

class LabelWriterThread(threading.Thread):
    def __init__(self, queue, dest_dir):
        super(LabelWriterThread, self).__init__()
        self.queue = queue
        self.daemon = True
        self.dest_dir = dest_dir

    def run(self):
        file_path = os.path.join(dest_dir, "manifest.txt")
        f = open(file_path, 'w')
        while True:
            f.write(self.queue.get() + "\n")
            self.queue.task_done()
        f.close()

class DownloadThread(threading.Thread):
    def __init__(self, url_queue, print_queue, classes, image_crop, dest_dir):
        super(DownloadThread, self).__init__()
        socket.setdefaulttimeout(URL_TIMEOUT)
        self.url_queue = url_queue
        self.classes = classes
        self.dest_dir = dest_dir
        self.daemon = True
        self.image_crop = image_crop
        self.print_queue = print_queue

    def run(self):
        while True:
            dict = self.url_queue.get()
            try:
                name = dict["url"].split('/')[-1]
                person_dir = os.path.join(self.dest_dir, dict["rel_dir"])

                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)

                dest_file = os.path.join(person_dir, name)
                self.download_url(dest_file, dict["url"])

                if os.path.isfile(dest_file):
                    if self.image_crop:
                        crop_dir = os.path.join(person_dir, "crop")

                        if not os.path.exists(crop_dir):
                            os.makedirs(crop_dir)

                        out_filename = os.path.join(crop_dir, 'crop_' + name)

                        self.crop_image(dest_file, out_filename, dict["crop_dims"])

                        if ESCAPE_SPACES:
                            out_filename = out_filename.replace(' ', '\ ')

                        self.print_queue.put(out_filename + '|0|0|0')
                    else:
                        if ESCAPE_SPACES:
                            dest_file = dest_file.replace(' ', '\ ')

                        self.print_queue.put(dest_file)

            except Exception, e:
                print("[%s] Error: %s" % (self.ident, e))

            self.url_queue.task_done()

    def download_url(self, dest_file, url):
        try:
            print("[%s] Downloading %s -> %s" % (self.ident, url, dest_file))
            u = urllib2.urlopen(url)
            with open(dest_file, "wb") as f:
                f.write(u.read())
            f.close()
        except urllib2.HTTPError, e:
            print("[%s] HTTP Error: %s %s" % (self.ident, e.code, url))
        except urllib2.URLError, e:
            print("[%s] URL Error: %s %s" % (self.ident, e.reason, url))


    def crop_image(self, dest_file, crop_file, crop_dims):
        print("[%s] Cropping %s -> %s" % (self.ident, dest_file, crop_file))
        c = crop_dims.split(',')
        img = Image.open(dest_file)
        img2 = img.crop((float(c[0]), float(c[1]), float(c[2]), float(c[3])))
        img2.save(crop_file)

def read_url_file(file_path):
    f = open(file_path)
    queue = Queue()
    classes = {}

    for line in f:
        if not line.startswith('#'):
            tokens = line.split('\t')
            queue.put({ "rel_dir": tokens[0], "url": tokens[2], "crop_dims": tokens[3]})
            if not tokens[0] in classes:
                classes[tokens[0]] = len(classes)

    f.close()
    return queue, classes

def write_class_file(classes, file):
    f = open(file, 'w')
    for key in classes:
        f.write(key + "\n")
    f.close()

if __name__ == "__main__":
    if len(sys.argv) <> 3:
        print("Usage: pub_fig_get.py <url_file> <dest_folder>")
        exit(0)

    url_file = sys.argv[1]
    dest_dir = sys.argv[2]
    class_file = os.path.join(dest_dir, "classes.txt")

    url_queue, classes = read_url_file(url_file)
    write_class_file(classes, class_file)
    print_queue = Queue()

    for i in range(NUM_THREADS):
        t = DownloadThread(url_queue, print_queue, classes, IMAGE_CROP, dest_dir)
        t.start()

    t = LabelWriterThread(print_queue, dest_dir)
    t.start()

    url_queue.join()
    print_queue.join()

