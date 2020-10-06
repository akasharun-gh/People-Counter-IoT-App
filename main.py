
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
'''
Note to reviewer: I have made use of the starter code provided by udacity for this course
and material taught in this course to help code this project. At times that I was stuck I 
have used Knowledge to find and understand possible solutions and implemented them by myself
in this project.
'''

import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def ssd_out_detect(frame, result, prob_threshold, width, height):
    
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame"""

    # current_count indiactes the count of people identified in the frame
    current_count=0
    for item in result[0][0]:
        # if confidence level is greater than the probability threshold, draw bounding box
        if item[2] > prob_threshold:
            xmin = int(item[3] * width)
            ymin = int(item[4] * height)
            xmax = int(item[5] * width)
            ymax = int(item[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            current_count += 1
    
    return frame, current_count


def inference_func(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    # Flag for the input image
    single_image_mode = False
    
    ### TODO: Load the model through `infer_network` ###
    input_shape = infer_network.load_model(args.model, args.device, args.cpu_extension)[1]

    ### TODO: Handle the input stream ###
    # Checks for webcam feed
    if args.input == 'CAM':
        input_stream = -1
    # Checks for image file 
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input
    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Input file not found"

    cap = cv2.VideoCapture(input_stream)
    
    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR: Cannot open video file")

    # get width and height of frame    
    width = cap.get(3)
    height = cap.get(4)
    
    net_input_shape = input_shape
    request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    person_detected = 0
    frame_count = 0
    frame_delay = 3
    end_detection = False
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
            
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        # change data format from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((1, 3, net_input_shape[2], net_input_shape[3]))

        ### TODO: Start asynchronous inference for specified request ###
        #im_shape = {'image_tensor': image,'image_info': image.shape[1:]}
        # store start time for inference
        inf_start = time.time()
        infer_network.exec_net(request_id, image)
        
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
            # calculate time when a person is detected
            detection_time = time.time() - inf_start      

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(request_id)

            ### TODO: Extract any desired stats from the results ###
            frame, current_count = ssd_out_detect(frame, result, prob_threshold, width, height)
            #log.error(current_count)
            #log.error(last_count)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            disp_inf_time = "Inference time: {:.2f}ms".format(detection_time * 1000)
            # Syntax: cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
            cv2.putText(frame, disp_inf_time, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (5, 5, 200), 1)
            
            # Check is new person has entered the video
            '''
            > end_detection is a boolean that is set to 'True' when a person is detected
            and is used to handle false restart of inference time if the person is 
            re-detected.
            > person_detected is used here as a tri-state variable that helps prevent
            multiple counts for the same person when detection accuracy is low.
            ''' 
            if current_count > last_count:
                if not end_detection:
                    start_time = time.time()
                    end_detection = True
                person_detected = 1
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
                
                
            # check for false double detection
            if current_count < last_count and last_count != 2:
                person_detected = -1
                
            # if person is not detected, increment frame_count   
            if person_detected == -1:
                frame_count += 1
            else:
                frame_count = 0
            '''
            wait till frame count becomes 20 after last detection to handle
            detection of same person multiple times.
            ''' 
            if frame_count == 20:
                # calculate duration of person in the frame
                # fame_delay is the delay caused by the wait after last detection; calculated by observation 
                duration = int(time.time() - start_time - frame_delay) 
                # Publish messages to MQTT server
                client.publish("person/duration", json.dumps({"duration": duration}))
                end_detection = False
                
            
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
            
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    inference_func(args, client)


if __name__ == '__main__':
    main()
    exit(0)
