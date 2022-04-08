import time
import cv2 #VS does not recognize this
#import sys #--> not used yet
import zmq #VS does not recognize this
import matplotlib.pylab as pit #not used yet**
import base64
import numpy as np


#vidcam = cv2.VideoCapture(0)
#if not (vidcam.isOpened()):
    #print("Video Camera Connection Issue")
print(2000)
port = "8083"
contextfunction = zmq.Context()
socket = contextfunction.socket(zmq.SUB) #PUB connection can only send
socket.connect("tcp://127.0.0.1:8083") #Should this be socket.connect() ?
socket.setsockopt_string(zmq.SUBSCRIBE, "")
print("socket connection worked")

while True:
    # read images from the camera
    #success, img = vidcam.read()
    #cv2.imshow("image", img) # display the image in a window named "image"
    #retval, img_jpg = cv2.imencode('.jpg', img)
    #reclatencystart = socket.recv_string()
    rectotal = socket.recv()
    rectime = rectotal[:20]
    rectime = rectime.decode()
    recpict = rectotal[20:]
    recpict = recpict #.encode()
    print(len(recpict))
    #pict = open("image", 'wb') ##wb --> binary format for writing
    decodeimage = bytearray(base64.b64decode(recpict))
    view_image = cv2.imdecode(np.frombuffer(decodeimage, np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("streaming images", view_image)
    print()
    reformatted_latency_start = base64.b64decode(rectime)
    Latency_time_end = time.time()
    elapsed_time = Latency_time_end - rectime
    print("Calculated Latency = ", elapsed_time)
    # if pressing "q" while the display window is active, "quit" the program by breaking the while loop
    #print framerate?
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
