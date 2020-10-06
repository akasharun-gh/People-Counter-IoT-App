# Project Write-Up : People Counter IoT Application using OpenVINO

## Explaining Custom Layers

The Model Optimizer of the OpenVINO Toolkit has a list of supported layers that it can optimize and run, but in certain rare cases there may be layers that are not supported by the Model Optimizer and these are classified as the Custom Layers. 
In order to convert the custom layer into IR, you have to register those layers as extensions to the Model Optimizer, which enables it to create a valid and optimized IR. In this project we use a cpu-extension in order to handle the unsupported custom layers. Some layers could be unsupported due to the hardware and in these cases the extensions add support to handle these layers and offload some of the work to the CPU.

## Model Selection

For this project I identified the ssd_mobilenet_v2_coco¬_ 2018_03_29 as the most suitable model for this application for reasons mentioned in the ‘Model Research’ section below. The steps involved in converting this model into an intermediate representation (IR) is as follows:
1. Download the model:
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

2. Extract the contents of the tar.gz file:
```
tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

3. Go into the directory of the extracted folder:
```
cd ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

4. Convert the tensorflow model to IR using the model optimizer python script:
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model
frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config                           --reverse_input_channel --tensorflow_use_custom_operations_config
/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json 
```

5. Command to run the application:
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.25 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```


## Comparing Model Performance

The chosen model is the ssd_mobilenet_v2_coco_2018_03_29 model. The comparisons of the model before and after conversion to IR were based on accuracy, model size and inference times.
1. Accuracy: On observation the accuracies of the model before and after conversion were similar and detection was  made with approximately the same confidence levels.
2. Size: Pre-conversion (frozen_inference_graph.pb) size = 69,688,296 bytes, Post-conversion (frozen_ inference_graph.xml + frozen_ inference_graph.bin) size = 67,272,876 bytes + 111,552 bytes = 67,384,428 bytes 
3. Inference time: Pre-converted tensorflow model approximate inference time was 93 ms. IR model had an approximate inference time of 69 ms.


## Assess Model Use Cases

Some potential use cases of the people counter app are,
1. At current times, a useful use-case would be in public buildings to ensure that the number of people do not exceed the maximum capacity in order to ensure a safe environment. With some modifications to the app, we could also check to see if people are maintaining a distance of six feet between them.
2. The application could be used in malls and schools to make sure no one is left behind before closing.
3. Another use-case could be at amusement park rides to count the number of people waiting in queues and estimate a waiting time for each ride, which can then be checked in a mobile application.

## Assess Effects on End User Needs

**Lighting:** is an important factor in detecting people in an environment. If the area under consideration is either too bright or too dark, it would make it difficult to identify edges in the frame, hence greatly reduce the accuracy of detection.
**Model Accuracy:** Models with higher accuracy are more beneficial, but usually more complex and resource heavy models provide more accurate results. If the user has resource constraints (depends on the edge device used), a model with lower accuracy could still work well with some additional modification to handle undetected frames as is done with this project.
**Image size and camera focal length:** The effects of these parameters greatly depend on the size and focal length of the images used during training of the model. Variations in these two parameters from the norm would reduce the accuracy of the model, leading to more missed detections.


## Model Research

In investigating potential people counter models, I tried each of the following three models:
**Model 1:** faster_rcnn_inception_v2_coco_2018_01_28
This model performed well in terms of detecting people in the video accurately, but the inference time for this model was really high, hence the video was slowed down quite a bit and could not be observed in real time.
Model link: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

Command to convert to IR:
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```

**Model 2:** ssd_inception_v2_coco_2018_01_28
This model was not very accurate in detecting people especially when they were stationary at the table. Even after reducing the probability threshold values to very low values, it was unable to detect the person in certain cases.
Model Link: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
Command to convert to IR:
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

**Model 3:** ssd_mobilenet_v2_coco¬_ 2018_03_29
This model was the most suitable for this application because it had fast inference times and it was more accurate than the ssd_inception_v2_coco_2018_01_28 model. With some modifications to the code to account for some lost frames, it worked really well, hence I used this model for this application. A probability threshold value of 0.25 was used for optimal results.
Model Link: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

| Model                                     | Size (bytes)        | Approx. Inference time (ms) |
| ------------------------------            |:-------------------:| ---------------------------:|
| faster_rcnn_inception_v2_coco_2018_01_28  | 53,229,380          | 852                         |
| ssd_inception_v2_coco_2018_01_28          | 100,074,252         | 156                         |
| ssd_mobilenet_v2_coco¬_ 2018_03_29        | 67,272,876          | 69                          |