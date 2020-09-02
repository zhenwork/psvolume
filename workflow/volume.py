import os,sys
PATH=os.path.dirname(__file__)
PATH=os.path.abspath(PATH+"./../")
if PATH not in sys.path:
    sys.path.append(PATH)

    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dset","--dset", help="dataset files", default=None, type=str) 
parser.add_argument("-proc","--proc", help="analysis list", default=None, type=str) 
parser.add_argument("-dir","--dir", help="work directory",  default="./", type=str) 
parser.add_argument("-tag","--tag", help="tag of workflow", default="test", type=str) 
args = parser.parse_args()

