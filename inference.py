#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.network = None
        self.core = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        ### TODO: Load the model ###
        model_xml = model
        model_weights = os.path.splitext(model_xml)[0] + ".bin"

        # Read IR as IENetwork
        self.network = IENetwork(model_xml, model_weights)
        ### TODO: Check for supported layers ###
        # Initialize the core
        self.core = IECore()

        supported_layers = self.core.query_network(network=self.network , device_name=device)

        ### Check for any unsupported layers, and let the user
        ### know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add to IECore.")
            #sys.exit(1)

        ### TODO: Add any necessary extensions ###
        # Add CPU extension if applicable
        if cpu_extension and 'CPU' in  device:
            self.core.add_extension(extension_path=cpu_extension, device_name="CPU")

        ### TODO: Return the loaded inference plugin ###
        self.exec_network = self.core.load_network(network=self.network, device_name=device, num_requests=1)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        ### Note: You may need to update the function parameters. ###
        return self.core, self.get_input_shape()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape


    def exec_net(self, request_id, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.infer_request = self.exec_network.start_async(request_id=request_id, inputs={self.input_blob:image})
        return self.exec_network

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        """
        Waits for the result to become available.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :return: Timeout value
        """
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id, output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        """
        Gives a list of results for the output layer of the network.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param output: Name of the output layer
        :return: Results for the specified request
        """
        if output:
            res = self.infer_request.outputs[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_blob]

        return res

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.exec_network
        del self.core

