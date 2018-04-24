import os
import numpy as np 
from mpidata import *
from fileManager import iFile
from imageMergeClient import ImageMerge
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--o", help="save folder", default=".", type=str)
parser.add_argument("-num","--num", help="num of images to process", default=-1, type=int)
args = parser.parse_args()
args.o = '/reg/data/ana04/users/zhensu/xpptut/volume'

zf = iFile()
if args.num==-1: num = zf.counterFile(args.o+'/mergeImage', title='.slice')
else: num = int(args.num)
num = int(args.num)

Vol = {}
Vol['volumeCenter'] = 60
Vol['volumeSampling'] = 1
Vol['volumeSize'] = 2*Vol['volumeCenter']+1
model3d = np.zeros([Vol['volumeSize']]*3)
weight  = np.zeros([Vol['volumeSize']]*3)
