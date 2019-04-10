import os
import actionCluster
import userActions
from mpi4py import MPI 
comm_rank = MPI.COMM_WORLD.Get_rank()
comm_size = MPI.COMM_WORLD.Get_size()


class ActionManager:
    def __init__(self, action=None, actionFile=None):
        self.action = action
        self.actionFile = actionFile
        self.actionStatus = "waiting"

    def start(self):
        if comm_size == 1:
            for action in actionList:
                actionName = action["actionName"]
                psvmParams = action["psvmParams"]
                tmpMaster = getattr(actionCluster, actionName)(psvmParams)
                tmpMaster.start(comm_rank, comm_size)
        elif comm_size > 1:
            for action in actionList:
                actionName = action["actionName"]
                psvmParams = action["psvmParams"]
                tmpMaster = getattr(actionCluster, actionName)(psvmParams)
                tmpMaster.start(comm_rank, comm_size)
        else:
            raise Exception("!! ERROR MPI RANK/SIZE")

    def checkStatus(self):
        return

    def 