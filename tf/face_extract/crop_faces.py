import sys
import threading
import io
import re
import os
from Queue import Queue
from PIL import Image
from google.cloud import vision

NUM_THREADS = 2
PROJECT_ID='replace-me-with-your-project-id'

class MetaWriterThread(threading.Thread):
    def __init__(self, queue, output_dir):
        super(MetaWriterThread, self).__init__()
        self.queue = queue
        self.daemon = True
        self.output_dir = output_dir

    def run(self):
        file_path = os.path.join(self.output_dir, "vision-manifest.txt")
        f = open(file_path, 'w')
        while True:
            f.write(self.queue.get() + "\n")
            self.queue.task_done()
        f.close()

class VisionThread(threading.Thread):
    def __init__(self, image_queue, print_queue, prepend_dir):
        super(VisionThread, self).__init__()
        self.image_queue = image_queue
        self.print_queue = print_queue
        self.prepend_dir = prepend_dir
        self.daemon = True
        self.client = vision.Client.from_service_account_json(os.path.join(sys.path[0], './vapi-acct.json'), PROJECT_ID)

    def run(self):
        while True:
            next_file = self.image_queue.get()
            filename = os.path.join(self.prepend_dir, next_file)
            print("[%s] Info: Opening file %s" % (self.ident, filename))

            try:
                with io.open(filename, 'rb') as image_file:
                    image = self.client.image(content=image_file.read())

                faces = image.detect_faces(limit=1)

                if faces != None and len(faces) > 0:
                    vertices = faces[0].fd_bounds.vertices

                    if vertices != None:
                        left = vertices[0].x_coordinate
                        top = vertices[0].y_coordinate
                        right = vertices[2].x_coordinate
                        bottom = vertices[2].y_coordinate

                        # crop image if all required verticies exists
                        if left and top and right and bottom:
                            file_dir = os.path.dirname(next_file)
                            base_file = os.path.basename(next_file)
                            crop_dir = os.path.join(file_dir, 'crop')

                            # create the output directory if it doesn't exist
                            if not os.path.exists(crop_dir):
                                print("[%s] Info: mkdir %s" % (self.ident, crop_dir))
                                os.makedirs(crop_dir)

                            original = Image.open(filename)
                            cropped = original.crop((left, top, right, bottom))
                            out_file = os.path.join(crop_dir, 'crop_' + base_file)

                            print("[%s] Info: Saving file %s" % (self.ident, out_file))
                            cropped.save(out_file)

                            # try and get face angle information
                            angles = faces[0].angles
                            pan = ''
                            roll = ''
                            tilt = ''

                            if angles != None:
                                pan = angles.pan
                                roll = angles.roll
                                tilt = angles.tilt

                            # write image details to data file
                            line = "%s|%s|%s|%s" % (out_file, angles.pan, angles.roll, angles.tilt)
                            self.print_queue.put(line)

                        else:
                            print("[%s] Error: Incomplete coordinates for %s" % (self.ident, filename))
                else:
                    print("[%s] Error: No face detected for %s" % (self.ident, filename))

            except Exception as e:
                print("[%s] Error: Exception occurred for %s: %s" % (self.ident, filename, str(e)))

            self.image_queue.task_done()

    def crop_image(self, filename):
        print("[%s] Cropping %s" % (self.ident, filename))

def read_file(file_path):
    f = open(file_path)
    queue = Queue()

    for line in f:
        if not line.startswith('#'):
            tokens = line.split('|')
            queue.put(tokens[0].rstrip('\n'))

    f.close()
    return queue

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: crop_faces.py <input_file> [prepend_dir]")
        exit(0)

    image_queue = read_file(sys.argv[1])
    prepend_dir = "./"

    if len(sys.argv) == 3:
        prepend_dir = sys.argv[2]

    print_queue = Queue()
    print_queue.put("#file|pan|roll|tilt")

    for i in range(NUM_THREADS):
        t = VisionThread(image_queue, print_queue, prepend_dir)
        t.start()

    t = MetaWriterThread(print_queue, os.path.dirname(sys.argv[1]))
    t.start()

    image_queue.join()
    print_queue.join()
