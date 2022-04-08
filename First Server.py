from typing import Counter
import cv2 #VS does not recognize this
#import sys #--> not used yet
import zmq #VS does not recognize this
import matplotlib.pylab as pit #not used yet**
import base64
import time

#Publishing to client:

vidcam = cv2.VideoCapture(0)
if not (vidcam.isOpened()):
    print("Video Camera Connection Issue")

port = "8083"
contextfunction = zmq.Context()
socket = contextfunction.socket(zmq.PUB)
socket.bind("tcp://*:8083")
Counter = 0
print("Counter initialized at", Counter)
time_start = time.perf_counter()


while True:
    Latency_Cycle_Start = '%.09f'%time.time() #client won't be able to access performance counter --> this is only relevant within the server.
    latlength = len(str(Latency_Cycle_Start)) #note that '' forces time.time() to string format
    print("Computer clock at ", Latency_Cycle_Start) #this line isn't necessary (and slows down server), just helpful for figuring things out.***
    print(latlength)
    #socket.send_string(str(Latency_Cycle_Start)) #problem here, because I'm using the same socket?
    Counter = Counter + 1 
    if Counter%60 == 0:
        time_end = time.perf_counter()
        elapsed_time = time_end - time_start
        print(elapsed_time, "seconds per frame") #How do you figure out what 'time last' is? --> 'time last' doesn't correlate to any functions.
        frame_rate = elapsed_time/60
        print("Frame rate =", frame_rate, "seconds per frame")
        time_start = time.perf_counter()
        #One idea--> make time last the last line in the while loop, which would be aproximately equal to the next time the while loop starts. 
    # read images from the camera
    success, img = vidcam.read()
    #cv2.imshow("server image", img) # display the image in a window named "image" *** Really don't need this here, commented out to avoid confusion with Client.
    retval, img_jpg = cv2.imencode('.jpg', img) 
    #bytes = bytearray(img_jpg)
    #imagestrng = base64.b64encode(img_jpg)
    imagestrng = base64.b64encode(img_jpg.flatten().tobytes())
    totalstring = Latency_Cycle_Start.encode() + imagestrng #.decode()
    print("latency length= ", len(Latency_Cycle_Start.encode()))
    print("total string length= ", len(totalstring))
    socket.send(totalstring) #was send_string()
    if cv2.waitKey(1) & 0xFF == ord('q'): # if pressing "q" while the display window 
        # is active, "quit" the program by breaking the while loop
        #note that this quit commanad doesn't work if the image pop up window is missing.
        break




