import os,sys
import time
import shlex
import datetime
import expTools
import fileManager
import utils as utils
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def actionObject(actionName):
    return getattr(sys.modules[__name__], actionName)

class ImageExtraction:
    def __init__(self, params={}):
        self.params = params
    def start(self):
        outDir = self.params["outDir"]
        fileList = self.params["fileList"]
        fileGXPARMS = self.params["fileGXPARMS"]
        expInfo = expTools.xdsIndex2psvm(fileGXPARMS)
        PsvolumeManager = fileManager.PsvolumeManager()

        for idx, fileName in enumerate(fileList):
            if idx%comm_size != comm_rank:
                continue
            dstName = outDir+"/"+os.path.basename(fileName).split(".")[0]+".slice"
            psvmParams = expTools.cbf2psvm(fileName)
            psvmParams = utils.mergeDict(old=psvmParams, new=fileGXPARMS)
            PsvolumeManager.psvm2h5py(psvmParams, dstName)
            psvmParams = None


class RemoveBadPixels:
    def __init__(self, params={}):
        self.params = params
    def start(self):
        outDir = self.params["outDir"]
        fileList = self.params["fileList"]
        dstName = outDir+"/"+os.path.basename(fileName)
        H5FileManager = fileManager.H5FileManager()
        PsvolumeManager = fileManager.PsvolumeManager()

        for idx, fileName in enumerate(fileList):
            if idx%comm_size != comm_rank:
                continue
            image = H5FileManager.h5reader(fileName, "image")
            



class PolarizationCorrection:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class SolidAngleCorrection:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class BlankSubtraction:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class ScalingFactor:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class ImageMerge:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class LaueSymmetrization:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class FriedelSymmetrization:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class RadialBackground:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

class SubtractRadialBackground:
    def __init__(self, params):
        self.params = params
    def start(self):
        return

