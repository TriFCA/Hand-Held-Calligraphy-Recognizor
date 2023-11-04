# success rate high, output to arduino
#
# 784 x 200 x 10 neural network mnist hand writing test
# original model was trained on a pc with 10k 28x28 samples
# the original samples are without much background noise
# then the model was transferred to a raspberry pi with v1 camera
# the output is sent via the GPIO serial IO interface to an arduino mega2560
# which in turn writes to a 480x320 tft lcd screen for display
# the arduino mega2560 is running a special version of terminal emulator
# for big 16x16 fonts and 32x50 numeric digits, with '#' symbol interpreted
# as embedding and unembedding of one 32x50 numerical digit at 160 x 160
#
# references:
#
# neural network model taken from "Make your own neural network", Tariq Rashid, 
#  Mar 31, 2016, CreateSpace
# image treated by cv2.Canny()
#
# Issues:
#
# target tracking is not treated, and is an issue left  to the operator and his eyesight
#
# suggest a future project to deal with tracking issue with a neural network
# if the recognition accuracy is to be vastly improved
#
# 

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import pickle

from picamera import PiCamera
from picamera.array import PiRGBArray
from time import sleep
import matplotlib.pyplot as plt
import cv2
import imutils

import serial

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes,outputnodes,learningrate,ihw,how):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = ihw
        self.who = how
        # use Sigmoid as excitation function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self,inputs_list,targets_list) :
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # training, update weights
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T,output_errors)
        self.who += self.lr*np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        np.transpose(hidden_outputs)
        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        pass

    def query(self,inputs_list) :
        inputs = np.array(inputs_list,ndmin=2).T
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def freeze_model(self,ihw,how):
        ihw = self.wih
        how = self.who
        pass

pass

# model parameters
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epoch = 5

arduino = serial.Serial('/dev/ttyAMA0',baudrate=9600)

# import the last working model
print("import last working model")
model_file_wih = open("model_wih.pickle",'rb')
model_data_wih = pickle.load(model_file_wih)
model_file_wih.close()
model_file_who = open("model_who.pickle",'rb')
model_data_who = pickle.load(model_file_who)
model_file_who.close()
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate,model_data_wih,model_data_who)

# turn on the camera, and capture an image
camera = PiCamera()
camera.resolution = (84,84)
camera.framerate = 20

# set up preview screen
# camera.start_preview()
# camera.preview_fullscreen = False
# camera.preview_window = (50,30,172,96)

# clear arduino display screen
arduino.write('\f'.encode())

# infinite loop
flag = 0
while (flag == 0) :

    # frame capture
    output = PiRGBArray(camera)
    camera.capture(output,'rgb')
    
    input_image_3 = cv2.resize(output.array,(84,84),interpolation=cv2.INTER_AREA)
    input_image_edge = cv2.Canny(input_image_3,20,130)
    input_frame = cv2.resize(input_image_edge,(28,28),interpolation=cv2.INTER_AREA)

    # cv2.imshow('input',input_frame)

    # display aim sight, rotate 270 degree
    display_image = cv2.resize(output.array,(280,280),interpolation=cv2.INTER_AREA)
    rotated_display_image = imutils.rotate(display_image,270)
    cv2.imshow('image',rotated_display_image)
    cv2.moveWindow('image',50,30)

    # display edge image in sight, also rotate 270 degree
    edge_image = cv2.resize(input_frame,(84,84),interpolation=cv2.INTER_AREA)
    rotated_edge_image = imutils.rotate(edge_image,270)
    cv2.imshow('edges',rotated_edge_image)
    cv2.moveWindow('edges',350,30)

    # do not reverse the color, just scale the input
    image_data = input_frame.reshape(784)
    image_data =(image_data/255.0*0.99)+0.01

    # query the model
    test_result = n.query(image_data)
    label = np.argmax(test_result)

    # now output the result to arduino mega for display

    arduino.write('Neural Network\r\n'.encode())
    arduino.write('Caligraph Recognition\r\n'.encode())
    arduino.write('I am looking at a: \r\n'.encode())
    arduino.write('\n\r'.encode())
    arduino.write('#'.encode())
    arduino.write(str(label).encode())
    arduino.write('#'.encode())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pass

cv2.destroyAllWindows()




