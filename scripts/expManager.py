import os
import time
import utils
import fileManager
import actionManager
aCM = actionManager.ActionManager()
oFM = fileManager.OtherFileManager()

class expManager:

    def __init__(self, actionFile=None, outDir="./"):
        self.expStatus = "waiting"
        self.actionFile = os.path.realpath(actionFile)
        self.outDir = os.path.realpath(outDir)
        self.psvolumeFile = self.outDir + "/psvolumeFile-%s.json"%utils.getTime()
        self.timeStamp = {utils.getTime() : "Start"}
        self.actionStatus = {}
        self.actionBackup = {}
        self.checkDirs(self.outDir)

    def start(self):
        self.expStatus = 'running'
        while True:
            if self.expStatus.lower() == "stop":
                self.updateStatus()
                break
            todoActions = self.getTodoActions()
            if len(todoActions) > 0:
                self.launchAction(todoActions)
                print "##### launched action: %s" % todoActions
            time.sleep(3)


    def to_dict(self):
        return self.__dict__


    def initialize(self, actionFile=None, outDir="./"):
        self.expStatus = "Waiting"
        self.actionFile = os.path.realpath(actionFile)
        self.outDir = os.path.realpath(outDir)
        self.psvolumeFile = self.outDir + "/psvolumeFile-%s.json"%utils.getTime()
        self.timeStamp[utils.getTime()] = "initialize"
        self.actionStatus = {}
        self.actionBackup = {}
        self.checkDirs(self.outDir)


    def updateStatus(self):
        while True:
            if oFM.save_json(self.actionStatus, self.psvolumeFile):
                break
            else: pass
        return True

    def checkDirs(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    def loadActions(self):
        while True:
            actions = oFM.load_json(self.actionFile)
            if actions: 
                return actions
            else: pass

    def getNextAction(self):
        
        if len(self.actionStatus) == 0:
            lastProcessedActionID = '00000'
        else:
            lastProcessedActionID = max(self.actionStatus)
        
        newActions = self.loadActions()
        todoActions = {}

        if len(newActions) == 0:
            return {}
        if lastProcessedActionID == max(newActions):
            return {}
        for actionID in sorted(newActions):
            if actionID <= lastProcessedActionID:
                continue
            else:
                todoActions[actionID] = newActions[actionID]
                break
        return todoActions



