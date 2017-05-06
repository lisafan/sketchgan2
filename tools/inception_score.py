import numpy as np
import os.path
import sys
import argparse
from scipy import ndimage
from inception_model import get_inception_score

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
a = parser.parse_args()

def main():
    input_dir = a.input_dir
    # for real photos
    #input_dir = '/a/data/grp1/rendered_256x256/256x256/photo/tx_000100000000'

    images = []
    # turn images in subdirectories of input_dir into numpy arrs
    for dirname in os.listdir(input_dir):
        for filename in os.listdir(input_dir+'/'+dirname):
            fn, ext = os.path.splitext(filename.lower())
        # remove endswith for real photos
        if fn.endswith('output') and (ext == ".jpg" or ext == ".png"):
            file_path = os.path.join(input_dir, dirname, filename)
            images.append(ndimage.imread(file_path))

    # Call this function with list of images. Each of elements should be a 
    # numpy array with values ranging from 0 to 255.
    mean, std = get_inception_score(images, splits=10)
    print('scores mean:', mean, ' scores std:', std)

main()
