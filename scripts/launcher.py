import os,sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dset","--dset","-d","--d", help="dataset files", default=None, type=str) 
parser.add_argument("-proc","--proc","-p","--p", help="analysis list", default=None, type=str) 
parser.add_argument("-dir","--dir","-d","--d", help="work directory", default="./", type=str) 
parser.add_argument("-tag","--tag", help="tag of the whole workflow", default="test", type=str) 
args = parser.parse_args()



from core.state import DataPoints
from core.action import ActManager
from core.workflow import WorkFlow

dset = DataPoints(args.dset)
proc = ActManager(args.proc)

workflow = WorkFlow(state=dset, action=proc, workdir=args.dir, tag=args.tag)
workflow.start(verbose=5)
workflow.dump()
print("workflow done")
