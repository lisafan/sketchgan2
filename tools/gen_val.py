# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import argparse
from scipy import ndimage
from collections import defaultdict
import csv

MODEL_DIR = '/a/data/grp1/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_generator_score(images, classes, cls_to_uid, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 100

  uid_to_cls = defaultdict(lambda:125)
  f = open('/a/data/grp1/class_id.csv')
  reader = csv.reader(f)
  for row in reader:
        uid_to_cls[row[2]] = int(row[0])

  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    print("Preds:",preds[0],preds.shape)
    
    idx = np.argsort(preds, axis=0)
    top_5s = idx[:,-5:]
    print("SHAPE", top_5s.shape)
    top_5s = np.array([[cls_to_uid[y] for y in x] for x in top_5s])
    print("SHAPE", top_5s.shape)
    #top_5s = [uid_to_cls[x] for x in top_5s]

    accuracy = 0.0
    top_five = 0.0
    for c, t_5 in zip(classes, top_5s):
	print(t_5[-1])
	if c in t_5:
		top_five += 1.0
	if t_5[-1] == c:
		accuracy += 1.0
    
    accuracy /= len(classes)
    top_five /= len(classes)

    #for i in range(splits):
    #  part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    #  kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    #  kl = np.mean(np.sum(kl, 1))
    #  scores.append(np.exp(kl))
    #return np.mean(scores), np.std(scores)
    return accuracy, top_five

def get_cls_to_uid():
  # File containing the mappings between class-number and uid. (Downloaded)
  path_uid_to_cls = "imagenet_2012_challenge_label_map_proto.pbtxt"
  cls_to_uid = {}

  # Read the uid-to-cls mappings from file.
  path = os.path.join(MODEL_DIR, path_uid_to_cls)
  print(path)
  with open(path, mode='r') as file:
      # Read all lines from the file.
            lines = file.readlines()
            for line in lines:
                # We assume the file is in the proper format,
                # so the following lines come in pairs. Other lines are ignored.
                if line.startswith("  target_class: "):
                    # This line must be the class-number as an integer.
                    # Split the line.
                    elements = line.split(": ")
                    # Get the class-number as an integer.
                    cls = int(elements[1])

                elif line.startswith("  target_class_string: "):
                    # This line must be the uid as a string.
                    # Split the line.
                    elements = line.split(": ")
                    # Get the uid as a string e.g. "n01494475"
                    uid = elements[1]
                    # Remove the enclosing "" from the string.
                    uid = uid[1:-2]
                    # Insert into the lookup-dicts for both ways between uid and cls
                    cls_to_uid[cls-1] = uid
  return cls_to_uid

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
a = parser.parse_args()

def main():
    if softmax is None:
	_init_inception()

    input_dir = a.input_dir
    # for real photos
    input_dir = '/a/data/grp1/rendered_256x256/256x256/photo/tx_000100000000'

    images = []
    classes = []
    i=0
    # turn images in subdirectories of input_dir into numpy arrs
    for dirname in os.listdir(input_dir):
	if i==20:
		break
	for filename in os.listdir(input_dir+'/'+dirname):
		print(dirname)
		i += 1
		if i == 20:
			break
        	fn, ext = os.path.splitext(filename.lower())
        	# remove endswith for real photos
        	if ext == ".jpg" or ext == ".png":
			#if fn.endswith('outputs') and (ext == ".jpg" or ext == ".png"):
            		file_path = os.path.join(input_dir, dirname,filename)
  			if not tf.gfile.Exists(file_path):
    				tf.logging.fatal('File does not exist %s', file_path)
  				images.append(tf.gfile.FastGFile(image, 'rb').read())
            		#images.append(ndimage.imread(file_path))
	    		#classes.append(int(fn.split('_')[0]))
			classes.append(fn.split('_')[0])
    print(classes)
    cls_to_uid = get_cls_to_uid()
    # Call this function with list of images. Each of elements should be a 
    # numpy array with values ranging from 0 to 255.
    accuracy, top_five = get_generator_score(images, classes, cls_to_uid)
    print('Top 1 Accuracy:', accuracy, ' Top 5 Accuracy:', top_five)

main()

