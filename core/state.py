import json
import os,sys


class State(object):
    def __init__(self, fname=None, _dict=None):
        if fname is not None:
            self.fromfile(fname)
        if _dict is not None:
            self.fromdict(_dict) 
            
    def __getattr__(self,name):
        return None
    
    def __cmp__(self,other):
        if not hasattr(other,"key"):
            return False
        return self.key == getattr(other, "key")
    
    def prev(self):
        if len(self.history)<2:
            return None
        return self.history[-2][1]
    
    def todict(self):
        out = self.__dict__.copy()
        return out
    
    def fromdict(self,_dict):
        self.__dict__.update(_dict)
        if not hasattr(self, history):
            self.history = []
            self.history.append((None, self))
            return 
        for idx in range(len(self.history)):
            self.history[idx] = (Action(_dict=self.history[idx][0]), State(_dict=self.history[idx][1])) 
        
    def tofile(self,fname):
        json.dump(self.todict(self.data), open(fname,"w"), indent=4)
    
    def fromfile(self,fname):
        self.fromdict(json.load(open(fname,"r")))
    