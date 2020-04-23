import json
import time
import copy
import os,sys
import pickle
import random
import inspect
import datetime
import subprocess
import numpy as np

print("import transaction:",datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


class ActionManager:
    """
    1. add action names
    2. search action objects
    3. run actions (here,term,bsub)
    4. check action status
    5. return receipt
    """
    def __init__(self,action=[],params=[],struct=None):
        self.action = action
        self.params = params
        self.status = []
        self.struct = struct
        self.receipt = None
    def append(self,data):
        ## data is dict/ActionMangaer
        ## data key or value length may not fit
        ## for example, x = {action:[action1,action2],params:parameters}
        ##    append should extend the params into list
        if isinstance(data,dict):
            for key in data.keys():
                ## receiver is a list
                if isinstance(getattr(self,key),(list,tuple)):
                    getattr(self,key).append(data[key])
                ## receiver is not a list
                elif len(self)==1:
                    setattr(self,key,[getattr(self,key),data[key]])
                else:
                    ## first expand the receivor to len(self)
                    setattr(self,key,[copy.deepcopy(getattr(self,key)) for _ in range(len(self))])
                    getattr(self,key).append(data[key])
        elif type(data) is ActionManager:
            self.append(data.__dict__)
        else:
            raise NotImplemented
    def extend(self,data):
        for x in data:
            self.append(x)
    def detect(self,actionobj=None,actionName=None):
        ## check whether the action exists
        if actionobj is not None:
            ## if you can add an object, then it exists already
            return True
        elif actionName is not None:
            paths = ['./actions.py']
            for path in paths:
                if os.path.isfile(path):
                    obj = imp.load_source('module.name', path)
                    if hasattr(obj,actionName):
                        return True
            print("!! action not found: ",str(action),str(type(action)))
            return False
        ## no inputs, meaning test everything inside the object
        marker = []
        for action in self.action:
            ## action is an object but not just the name
            if isinstance(action,str): 
                marker.append(self.detect(actionName=action))
                continue
            marker.append(self.detect(actionobj=action))
        return all(marker)
    def start(self):
        order = self.getorder()
        self.status = ["waiting"]*self.__len__()
        ## order={0:set({1,2,3}),...}
        while not self.finish():
            nextAction = self.getNext()
            hook = nextAction.run()
            self.update({"status":"running"})
            time.sleep(10)
        return True
    def receipt(self):
        return 
    def finish(self):
        return 
    def getNext(self):
        ## returns: [], [obj], [obj1,obj2,...]
        if len(self.action) == 0:
            return []
        ## get which action is Done
        doneidx = []
        for actionidx,status in enumerate(self.status):
            if status.lower() == "done":
                doneidx.append(actionidx)
        nextAction = []
        for actionidx,prepare in self.getorder().items():
            if (len(prepare) == 0) and actionidx not in doneidx:
                nextAction.append(actionidx)
                continue
            alldone = all([preidx in doneidx for preidx in prepare])
            if alldone:
                nextAction.append(actionidx)
        return nextAction
    def update(self):
        ## get job commuication, update which one is done
        if (self.status) == 0:
            self.status = ["waiting"] * len(self)
        for actionidx, actionobj in enumerate(self.action):
            self.jobstatus(actionobj)
        return
    def jobstatus(self,actionobj):
        
        return # ["waiting","running","done","failed"]
    def __getitem__(self,item):
        out = ActionManager() 
        for key,data in self.__dict__.items():
            if isinstance(data,(list,tuple)) and len(data)>1:
                setattr(out,key,data[item])
            else:
                setattr(out,key,data)
        return out
    def __setitem__(self,item,data):
        ## has many data to set, in the format of list(dict)
        ##     another similar one is dict(list) format, means the same
        if isinstance(data,(list,tuple)) and len(data)>1:
            for idx in range(len(data)):
                self.__setitem__(idx,data[idx])
        elif isinstance(data,dict):
            ## check whether inner keys have longer length
            nset = 0
            if "action" in data and isinstance(data["action"],(list,tuple)):
                nset = max(nset,len(data["action"]))
            if "params" in data and isinstance(data["params"],(list,tuple)):
                nset = max(nset,len(data["params"]))
            if nset == 0:
                pass
            elif nset == 1:
                ## data has only one action
                ## self may have more actions/one action 
                for key in data.keys():
                    if isinstance(data[key],(list,tuple)):
                        ## data is a length 1 list
                        tmpData = data[key][0]
                    else:
                        ## data is not a list
                        tmpData = data[key]
                    
                    if isinstance(getattr(self,key),(list,tuple)):
                        ## it means the receivor is list
                        getattr(self,key)[item] = tmpData
                    else:
                        ## it means the receivor is a number
                        setattr(self,key,[copy.deepcopy(getattr(self,key)) for _ in range(len(self))])
                        getattr(self,key)[item] = tmpData
                        
            ## have more than 1 data to set
            for key in data.keys():
                if isinstance(data[key],(list,tuple)):
                    ## data is a length 1 list
                    tmpData = data[key][item]
                else:
                    ## data is not a list
                    tmpData = data[key]
                if isinstance(getattr(self,key),(list,tuple)):
                    ## it means the receivor is list
                    getattr(self,key)[item] = tmpData
                else:
                    ## it means the receivor is a number
                    setattr(self,key,[copy.deepcopy(getattr(self,key)) for _ in range(len(self))])
                    getattr(self,key)[item] = tmpData
        elif type(data) is ActionManager:
            ## check whether inner class has longer length
            self.__setitem__(item,data.__dict__)
        else:
            raise NotImplemented
    def __call__(self):
        return out
    def todict(self):
        return out
    def release(self,fname=None,folder=None,header=None):
        return 
    def getorder(self,struct=None):
        struct = struct or self.struct
        numbers = set()
        for i in str(struct):
            if i.isdigit():
                numbers.add(int(i))
        order = dict((i, set([])) for i in numbers)
        def refresh(struct,order):
            for logic in struct:
                X,Y = logic
                if not isinstance(Y,(list,tuple)):
                    Y = [Y]
                if not isinstance(X,(list,tuple)):
                    X = [X]
                for y in Y:
                    for x in X:
                        order[y].add(x)
                        order[y] = order[y].union(order[x])
            return order
        oldorder = order.copy()
        while oldorder != refresh(struct,order):
            oldorder = order.copy()
            continue
        return order