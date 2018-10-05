#!/usr/bin/env python3

# https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/MultiStickSSDwithRealSense.py

#------ parameters --------#
FPS = 60
IMGSIZE = 224
#IMGSIZE = 160

SHOW_IMGS = True
VERBOSE = False
#--------------------------#




import mvnc.mvncapi as mvnc
import numpy as np
from PIL import Image
import cv2
import time, sys, os
import glob
import argparse
import multiprocessing as mp


parser = argparse.ArgumentParser(description="RMScript")
parser.add_argument('-f', help="FPS [1,60]", type=int)
parser.add_argument('-s', help="SIZE [160 or 224]", type=int)
parser.add_argument('-v', help="SHOW IMAGE [T or F]")
args = parser.parse_args()
if args.f:
    FPS = args.f
if args.s:
    IMGSIZE = args.s
if args.v and args.v=='T':
    SHOW_IMGS = True



GRAPH_FILE = '../graphs/mbnet224_1.graph'
if IMGSIZE == 160:
    GRAPH_FILE = '../graphs/mbnet160_05.graph'


CATEGORIES_FILE = './categories.txt'
with open(CATEGORIES_FILE, 'r') as f:
    categories = f.read().split('\n')
print("Graph file from", GRAPH_FILE)


TMP_IMG_NAME = './tmpimage.png'


class MeasureTime(object):
    def __init__(self, label, verbose):
        self.start = time.time()
        self.label = label
        self.verbose = verbose
    def stop_ms(self):
        return (time.time() - self.start)*1000
    def show(self):
        if self.verbose:
            print(self.label + "{:.3f}".format(self.stop_ms()), end=' : ')


def camThread(results, frameBuffer):
    res = ''
    window_name = "Frame"

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Couldn't open Camera")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, FPS)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        s, color_image = cam.read()
        if not s:
            continue
        color_image = cv2.resize(color_image, (IMGSIZE, IMGSIZE)) # TODO BGR2RGB 1/255
        if frameBuffer.full():
            frameBuffer.get()
        frameBuffer.put(color_image.copy())

        if not results.empty():
            res = results.get(False)
        cv2.putText(color_image, res, (0,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3, cv2.LINE_AA)

        cv2.imshow(window_name, color_image)
        cv2.waitKey(1)


def prepare_img(cap):
    tmr = MeasureTime("Prep", verbose=VERBOSE)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (IMGSIZE,IMGSIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.asarray(frame) * (1.0/255.0)
    tmr.show()
    return frame

def predict(graph, frame):
    tmr = MeasureTime("Pred", verbose=VERBOSE)
    # Write the tensor to the input_fifo and queue an inference
    graph.LoadTensor(frame.astype(np.float16), None)
    output, userobj = graph.GetResult()
    predict_resl = np.argmax(output)
    tmr.show()
    return predict_resl, output

def show_img(img, text):
    tmr = MeasureTime("Post", verbose=VERBOSE)
    if SHOW_IMGS:
        cv2.putText(img, text, (0,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3, cv2.LINE_AA)
        cv2.imshow('Frame', img)
        cv2.waitKey(1)
    tmr.show()


def main(device, graph, cap):
    print("Start predicting ...")

    while True:
        start = time.time()

        img = prepare_img(cap)  # Prepare Image

        result, output = predict(graph, img)    # Predict Image

        text = "{} {:.3f}".format(categories[result], output[result]) 
        print(text, end=' : ')

        show_img(img, text)     # Show Image

        totl = time.time() - start
        print("Totl{:.3f} : FPS{:.2f}".format(totl*1000, 1/totl), end=' : ')
        print('')

    return output

def mlt_main():
    print("Start mlt_main...")
    processes = []
    try:
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        p = mp.Process(target=camThread, args=(results, frameBuffer), daemon=True)
        p.start()
        processes.append(p)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        print("Finished")


if __name__ == '__main__':
    mlt_main()


if not __name__ == '__main__':
    devices = mvnc.EnumerateDevices()
    print("%d devices found" % len(devices))
    device = mvnc.Device(devices[0])
    device.OpenDevice()

    cap = cv2.VideoCapture(0)
    print("camera open", cap.isOpened())
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)#IMGSIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)#IMGSIZE)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Load graph file data
    with open(GRAPH_FILE, 'rb') as f:
        graph_file_buffer = f.read()

    # Initialize a Graph object
    graph = device.AllocateGraph(graph_file_buffer)

    try:
        main(device, graph, cap)
    except KeyboardInterrupt:
        print('exit')
        graph.DeallocateGraph()
        device.CloseDevice()
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
