import os,sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dset","--dset", help="dataset files", default=None, type=str) 
parser.add_argument("-proc","--proc", help="analysis list", default=None, type=str) 
parser.add_argument("-dir","--dir", help="work directory",  default="./", type=str) 
parser.add_argument("-tag","--tag", help="tag of workflow", default="test", type=str) 
args = parser.parse_args()


from diffuse.fconvert import Fconvert
from core.workflow import WorkFlow

state  = Fconvert.state(args.dset)
action = Fconvert.action(args.proc)

workflow = WorkFlow(state=state,actions=action,workdir=args.dir,tag=args.tag)
workflow.start(verbose=3)
workflow.dump()
print("workflow done")

