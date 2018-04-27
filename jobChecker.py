import time
import os, sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--i", help="save folder", default=".", type=str)
args = parser.parse_args()

if args.i == '.' or not os.path.exists(args.i): 
	raise Exception('no such file')