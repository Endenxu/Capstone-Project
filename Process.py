import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import threading
import time
import serial

img_height = 180
img_width = 180
cam = None
frame = None
img = None
last_recorded_time = None

def ProcessImage():
    global model, class_names, img, frame, ser
    print("Starting recognition")
    img = keras.preprocessing.image.load_img(
        "pendingimage.jpg", target_size=(img_height, img_width)
    )
    
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
        
    
    if class_names[np.argmax(score)] == 'no object':
        print("No Object. Sending no value to Arduino")         
    elif class_names[np.argmax(score)] == 'Class A':
        ser.write(bytes("1", 'utf-8'))
        print("This image most likely belongs to Class A with a {:.2f} percent confidence. Sending value {} to Arduino"
              .format(100 * np.max(score), np.argmax(score) + 1))         
    elif class_names[np.argmax(score)] == 'Class B':
        ser.write(bytes("2", 'utf-8'))
        print("This image most likely belongs to Class B with a {:.2f} percent confidence. Sending value {} to Arduino"
              .format(100 * np.max(score), np.argmax(score) + 1))
    elif class_names[np.argmax(score)] == 'Class C':
        ser.write(bytes("3", 'utf-8'))
        print("This image most likely belongs to Class C with a {:.2f} percent confidence. Sending value {} to Arduino"
              .format(100 * np.max(score), np.argmax(score) + 1))

def StartCamera():
    global cam, frame, last_recorded_time
    cam = cv2.VideoCapture(1)
    cv2.namedWindow("Camera 1 - Press ESC to Exit, Press Space to manually detect")
    img_counter = 0
    last_recorded_time = time.time()
    start_time = time.time()
    frame_count = 0
    
    while True:
        
        curr_time = time.time()
        ret, frame = cam.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)  
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,            
            minDist=50,      
            param1=50,        
            param2=30,        
            minRadius=10,     
            maxRadius=30    
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, radius = circle[0], circle[1], circle[2]
                cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

        
        fps = frame_count / (curr_time - start_time) if curr_time != start_time else 0.0
       
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Camera 1 - Press ESC to Exit, Press Space to manually detect", frame)
        
        if curr_time - last_recorded_time >= 3.0:
            img_name = "pendingimage.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            ProcessImage()
            last_recorded_time = curr_time
            
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            img_name = "pendingimage.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            ProcessImage()
    
        frame_count += 1
    
    cam.release()
    cv2.destroyAllWindows()

model = keras.models.load_model('./')
class_names = ['Class A', 'Class B', 'Class C', 'no object']
ser = serial.Serial('COM3', 9600)
StartCamera()
