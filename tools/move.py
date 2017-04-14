import os
import argparse
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--dir", default="/a/h/lfan01/sketchgan2/test/images")
a = parser.parse_args()

def main():
	img_path = '/a/h/lfan01/sketchgan2/test/images/'
	for root, dirs, files in os.walk(a.dir):
		for f in files:
			batch = f.split('-')[0]
			if not os.path.exists(img_path + batch):
				os.mkdir(img_path + batch)
				os.chmod(img_path + batch, 0771)
			shutil.move(os.path.join(root, f), img_path+batch)
			os.chmod(os.path.join(img_path+batch,f), 0771)

main()		
