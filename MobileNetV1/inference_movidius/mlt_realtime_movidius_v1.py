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


def camThread(results, frameBuffer):
    print("camThread start")
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
        color_image = cv2.resize(color_image, (IMGSIZE, IMGSIZE))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        #color_image = np.asarray(color_image) * (1.0/255.0)
        if frameBuffer.full():
            frameBuffer.get()
        frameBuffer.put(color_image.copy())

        if not results.empty():
            output = results.get(False)
            #prep = np.argmax(output)
            #res = "{cate}: {prob}".format(categories[prep], output[prep])
            res = "{cate}".format(output)
        cv2.putText(color_image, res, (0,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3, cv2.LINE_AA)

        cv2.imshow(window_name, color_image)
        cv2.waitKey(1)

def inferenceThread(results, frameBuffer, devnum):
    print("inferenceThread start")

    # Load graph file data
    with open(GRAPH_FILE, 'rb') as f:
        graph_file_buffer = f.read()

    device_name = mvnc.EnumerateDevices()[devnum]
    device = mvnc.Device(device_name)
    print("open")
    device.OpenDevice()

    graph = device.AllocateGraph(graph_file_buffer)

    while True:
        if frameBuffer.empty():
            continue

        frame = frameBuffer.get()
        graph.LoadTensor(frame.astype(np.float16), None)
        output, userobj = graph.GetResult()
        predict_resl = np.argmax(output)

        results.put(categories[predict_resl])



def main():
    processes = []
    try:
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        p = mp.Process(target=camThread, args=(results, frameBuffer), daemon=True)
        p.start()
        processes.append(p)

        n_devices = len(mvnc.EnumerateDevices())
        print("{} devices found".format(n_devices))
        for devnum in range(n_devices):
            p = mp.Process(target=inferenceThread, args=(results, frameBuffer, devnum), daemon=True)
            p.start()
            processes.append(p)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
        print("Finished")


if __name__ == '__main__':
    main()
