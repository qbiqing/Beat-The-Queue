import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import requests

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './models/exported/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './data/label_map.pbtxt'

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in ['num_detections', 'detection_boxes', 'detection_scores', \
			'detection_classes', 'detection_masks']:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						tensor_name)
			if 'detection_masks' in tensor_dict:
		    	# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
		        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)

			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			# Run inference
			output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
			    output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict

# MAIN
cap = cv2.VideoCapture(0)
outfile = open('output.txt', 'w')
lcount = 1

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	# Actual detection.
	output_dict = run_inference_for_single_image(frame, detection_graph)
	# Visualization of the results of a detection.
	image, count = vis_util.visualize_boxes_and_labels_on_image_array(
	    frame,
	    output_dict['detection_boxes'],
	    output_dict['detection_classes'],
	    output_dict['detection_scores'],
	    category_index,
	    instance_masks=output_dict.get('detection_masks'),
	    use_normalized_coordinates=True,
	    line_thickness=8)
	# print(count)
	cv2.putText(frame, "No. of people: " + str(count), (20, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))
	cv2.imshow("frame", frame)
	#print(count)

	if lcount == 0:
		outfile.write(str(count) + " ")
		outfile.flush()
		#r = requests.post("http://localhost:8080/test", data={'number': count, 'type': 'issue', 'action': 'show'})
		#sys.stdout.flush()
	lcount +=1
	if lcount == 5:
		lcount =0
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




