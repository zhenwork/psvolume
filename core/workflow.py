import json 
import os,sys
from core.actmaster import ActManager

"""
- ActManager(state,one action) launch one action and return the final state
- Workflow keeps the current state, status of each action
"""
class WorkFlow:
    def __init__(self,state=None,action=[],workdir="./",tag=None):
        self.tag = tag
        self.state = state 
        self.workdir = os.path.realpath(workdir)

        self.action = action
        if not isinstance(action,list):
            self.action = [action]

    def start(self):
        self.prepare() 
        for idx,actpack in enumerate(self.action):
            master = ActMaster(state=self.state.copy(), action=actpack, workdir=self.workdir, tag= )
            master.start().wait(refresh=1)
            self.status[idx] = master.status()
            if self.status[idx] != "done":
                break  
            self.state = master.returns().copy()

    def prepare(self):
        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir) 
        if not hasattr(self, "status"):
            self.status = [None for _ in self.action]

    def todict(self):
        return self.__dict__

    def fromdict(self, _dict):
        self.__dict__.update(_dict) 
        for idx in range(len(self.history)):
            self.history[idx] = (Action(_dict=self.history[idx][0]), State(_dict=self.history[idx][1])) 

    def tofile(self, fname):
        json.dump(self.todict(), open(fname,"w"), indent=4)

    def fromfile(self, fname):
        self.fromdict(json.load(open(fname,"r")))

