#!/usr/bin/env python3

#------ parameters --------#
FPS = 30
#IMGSIZE = 224
IMGSIZE = 160

SHOW_IMGS = True
onRPI = False
#--------------------------#




from mvnc import mvncapi
import numpy as np
from PIL import Image
import cv2
import time, sys, os
import glob
import argparse


parser = argparse.ArgumentParser(description="RMv2Script")
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



def predict(device, graph, cap, input_fifo, output_fifo):
    print("Start predicting ...")

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not onRPI:
            frame = cv2.resize(frame, (IMGSIZE,IMGSIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame) * (1.0/255.0)
        #frame = np.reshape(frame, (1, IMGSIZE, IMGSIZE, 3))
        frame = frame.astype(np.float32)
        print("Input shape", frame.shape)
        time1 = time.time()
        print("Prep{:.3f}".format((time1 - start)*1000), end=' : ')

        # Write the tensor to the input_fifo and queue an inference
        graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, frame, 'user object')
        output, user_obj = output_fifo.read_elem()
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
    # Get device list
    device_list = mvncapi.enumerate_devices()
    print("Found {} Devices : ".format(len(device_list)))

    # Open Device
    device = mvncapi.Device(device_list[0])
    device.open()
    print(device.get_option(mvncapi.DeviceOption.RO_DEVICE_NAME))

    # Open Camera & Config
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMGSIZE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMGSIZE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Load graph file data
    with open(GRAPH_FILE, 'rb') as f:
        graph_buffer = f.read()

    # Initialize a Graph object
    graph = mvncapi.Graph('graph1')

    # Allocate the graph to the device and create input and output Fifos
    input_fifo, output_fifio = graph.allocate_with_fifos(device, graph_buffer)

    try:
        predict(device, graph, cap, input_fifo, output_fifio)
    except KeyboardInterrupt:
        print('exit')
        input_fifo.destroy()
        output_fifo.destroy()
        graph.destroy()
        device.close()
        device.destroy()
        cap.release()
        sys.exit()
