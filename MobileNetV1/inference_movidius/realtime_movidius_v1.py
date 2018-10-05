#!/usr/bin/env python3

#------ parameters --------#
FPS = 10
IMGSIZE = 224
#IMGSIZE = 160

SHOW_IMGS = False
#--------------------------#




import mvnc.mvncapi as mvnc
import numpy as np
from PIL import Image
import cv2
import time, sys, os
import glob
import argparse


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



IMAGENET1000 = False
if IMAGENET1000:
    print("Use imagenet")
    import cate_imagenet1000
    categories = cate_imagenet1000.categories
    GRAPH_FILE = './models/mobilenet/graph'
else:
    CATEGORIES_FILE = './categories.txt'
    with open(CATEGORIES_FILE, 'r') as f:
        categories = f.read().split('\n')
    print("Graph file from", GRAPH_FILE)


TMP_IMG_NAME = './tmpimage.png'

def predict(device, graph, cap):
    print("Start predicting ...")


    while True:
        start = time.time()
        ret, frame = cap.read()
        frame = cv2.resize(frame, (IMGSIZE,IMGSIZE))
        print("shape", frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame) * (1.0/255.0)
        #frame = np.clip(frame, 0.0, 0.6)
        #print(frame.max(), round(frame.min(), 3), end=' ')
        time1 = time.time()
        print("Prep{:.3f}".format((time1 - start)*1000), end=' : ')

        # Write the tensor to the input_fifo and queue an inference
        graph.LoadTensor(frame.astype(np.float16), None)
        output, userobj = graph.GetResult()
        predict = np.argmax(output)

        time2 = time.time()
        print("Pred{:.4f}".format((time2-time1)*1000), end=' : ')

        text = "{} {:.3f}".format(categories[predict], output[predict])
        print(text, end=' : ')

        if SHOW_IMGS:
            cv2.putText(frame, text, (0,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3, cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            k = cv2.waitKey(1)
            if k == 27:
                break

        stop = time.time()
        print("Post{:.3f}".format((stop-time2)*1000), end=' : ')
        totl = stop-start
        print("Totl{:.3f}".format(totl*1000), end=' : ')
        print("FPS{:.2f}".format(1/totl), end=' : ')
        print('')

    cv2.destroyAllWindows()


    return output

if __name__ == '__main__':
    devices = mvnc.EnumerateDevices()
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
        predict(device, graph, cap)
    except KeyboardInterrupt:
        print('exit')
        graph.DeallocateGraph()
        device.CloseDevice()
        cap.release()
        sys.exit()
