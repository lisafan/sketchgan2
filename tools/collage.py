import os
import argparse
import shutil
import math
import numpy as np
from scipy import misc

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", required=True)
parser.add_argument("--output_dir", default="/a/h/lfan01/sketchgan2/test/images/collages")
a = parser.parse_args()

def collage(paths, root, batch, t):
	size = int(math.ceil(np.sqrt(len(paths))))
	imgs = [misc.imread(path) for path in paths]
	print "Found ", len(imgs), " images"
	
	blank = np.ones_like(imgs[0]) * 255
	imgs.extend(blank for i in range(size**2 - len(paths)))

	rows = [imgs[i:i+size] for i in range(0,len(imgs),size)]
	rows = [np.concatenate(row, axis=1) for row in rows]
	grid = np.concatenate(rows,axis=0)
	
	print "Saving to: ", os.path.join(root, batch+'_'+t+'_collage.png')
	misc.imsave(os.path.join(root, batch+'_'+t+'_collage.png'), grid)
	os.chmod(os.path.join(root, batch+'_'+t+'_collage.png'), 0771)

def main():
	inputs = []
	outputs = []
	targets = []

	for rt, dirs, files in os.walk(a.input_dir):
		root = rt
		batch = files[0].split('-')[0]
		for f in files:
			t = f.split('-')[-1]
			if t == 'inputs.png':
				inputs.append(os.path.join(root,f))
			if t == 'outputs.png':
				outputs.append(os.path.join(root,f))
			if t == 'targets.png':
				targets.append(os.path.join(root,f))

	inputs.sort()
	outputs.sort()
	targets.sort()

	collage(inputs, a.output_dir, batch, 'inputs')
	collage(outputs, a.output_dir, batch, 'outputs')
	collage(targets, a.output_dir, batch, 'targets')


main()
