import os
import numpy as np 
from userScript import *
from fileManager import *
from imageMergeClient import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
parser.add_argument("-box","--box", help="spot size", default=2, type=int)
args = parser.parse_args()


zf = iFile()
zio = IOsystem()

if ( not os.path.isfile(args.i) ) or (args.i == '.'):
	raise Exception('### no such file ... ')
	

print "### Reading the image: "+args.i
image = zf.h5reader(args.i, 'image')
Geo = zio.get_image_info(args.i)

print "### Processing the image: "+args.i
image = RemoveBragg(image, Geo, box=int(args.box))


fsave = args.i.split('/')[-1] + '-no-bragg'
print 'save file: '+fsave
zf.h5writer(fsave, 'image', image)