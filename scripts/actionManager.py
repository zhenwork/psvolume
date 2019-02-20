import os
import subprocess
import actionCluster
import userActions
import launchTools

class ActionManager:
    def __init__(self, action):
        self.actionID = action.keys()[0]
        self.jobID = None
        self.actionStatus = "waiting"

    def start(self):
        mode = self.

    def checkStatus(self, jobID=None, jobName=None):
        if jobID is not None:
            
    def launchAction(self, ):
        if len(todoActions) == 0:
            return True
        else:
            try:
                p = 
                p.start()
                p = None
                return True 
            except Exception as err: 
                print(err)
                return False